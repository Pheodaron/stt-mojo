"""HTTP adapter around the MAX model, independent of max serve's LLM scheduler."""

import hmac
import io
import json
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from .audio import FeatureExtractor, resample_mono, speaker_segments
from .model import NemotronDiarization
from .weights import resolve_checkpoint


class FeatureChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    input_features: list[list[float]] = Field(min_length=1, max_length=4096)
    attention_mask: list[bool] | None = None
    num_lookahead_frames: int = Field(default=0, ge=0, strict=True)
    num_lookback_frames: int = Field(default=0, ge=0, strict=True)
    is_last: bool = False


def create_app(
    model_path=None,
    *,
    revision=None,
    api_key=None,
    model=None,
    frontend=None,
    max_upload_bytes=64 * 1024 * 1024,
    max_audio_seconds=600,
    max_sessions=32,
    session_ttl=900,
):
    if model is None:
        path = resolve_checkpoint(model_path, revision)
        model = NemotronDiarization.from_pretrained(path)
        processor = path / "processor_config.json"
        frontend = (
            FeatureExtractor.from_processor_config(json.loads(processor.read_text()))
            if processor.exists()
            else FeatureExtractor()
        )
    if frontend is None:
        frontend = FeatureExtractor(feature_size=model.config.audio_config.num_mel_bins)
    if frontend.feature_size != model.config.audio_config.num_mel_bins:
        raise ValueError("Frontend mel count disagrees with checkpoint")
    if min(max_upload_bytes, max_audio_seconds, max_sessions, session_ttl) <= 0:
        raise ValueError("Service limits must be positive")
    lock = threading.Lock()
    sessions = OrderedDict()
    bearer = HTTPBearer(auto_error=False)

    def authenticate(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if api_key is not None and (
            credentials is None or not hmac.compare_digest(credentials.credentials, api_key)
        ):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="Nemotron 3 Diarization on MAX", version="0.1.0")

    def expire():
        now = time.monotonic()
        for key in list(sessions):
            if now - sessions[key][1] >= session_ttl:
                del sessions[key]

    @app.get("/healthz")
    def health():
        return {"status": "ready", "backend": "modular-max", "device": "cpu", "dtype": "float32"}

    @app.post("/v1/audio/diarizations", dependencies=[Depends(authenticate)])
    def diarize(file: UploadFile = File(...), threshold: float = Query(0.5, ge=0, le=1)):
        content = file.file.read(max_upload_bytes + 1)
        if len(content) > max_upload_bytes:
            raise HTTPException(413, "Audio upload exceeds byte limit")
        try:
            with sf.SoundFile(io.BytesIO(content)) as sound:
                duration = sound.frames / sound.samplerate
                if duration > max_audio_seconds:
                    raise HTTPException(413, "Audio exceeds duration limit")
                if sound.channels > 2 or sound.samplerate < 8000 or sound.samplerate > 192000:
                    raise ValueError("Expected mono/stereo audio sampled at 8–192 kHz")
                audio = resample_mono(
                    sound.read(dtype="float32", always_2d=True),
                    sound.samplerate,
                    frontend.sampling_rate,
                )
            features = frontend(audio, sampling_rate=frontend.sampling_rate)
        except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
            raise HTTPException(422, f"Invalid audio: {exc}") from exc
        with lock:
            output = model(features.input_features, features.attention_mask)
        return {
            "duration": duration,
            "frame_seconds": frontend.hop_length / frontend.sampling_rate,
            "segments": speaker_segments(
                output.probabilities[0],
                threshold=threshold,
                frame_seconds=frontend.hop_length / frontend.sampling_rate,
                attention_mask=features.attention_mask[0],
                duration=duration,
            ),
        }

    @app.post("/v1/diarization/sessions", status_code=201, dependencies=[Depends(authenticate)])
    def new_session():
        with lock:
            expire()
            if len(sessions) >= max_sessions:
                raise HTTPException(429, "Streaming session limit reached")
            key = str(uuid.uuid4())
            sessions[key] = (model.new_cache(), time.monotonic(), 0)
        return {
            "session_id": key,
            "mel_bins": model.config.audio_config.num_mel_bins,
            "subsampling_factor": model.config.audio_config.subsampling_factor,
            "frame_seconds": frontend.hop_length / frontend.sampling_rate,
        }

    @app.post("/v1/diarization/sessions/{session_id}", dependencies=[Depends(authenticate)])
    def chunk(session_id: str, request: FeatureChunk):
        try:
            features = np.asarray(request.input_features, dtype=np.float32)[None]
            if (
                features.shape[-1] != model.config.audio_config.num_mel_bins
                or not np.isfinite(features).all()
            ):
                raise ValueError("Invalid mel dimension or non-finite features")
            mask = (
                None if request.attention_mask is None else np.asarray(request.attention_mask)[None]
            )
            if request.is_last and request.num_lookahead_frames:
                raise ValueError("The last chunk must not contain look-ahead")
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        with lock:
            expire()
            if session_id not in sessions:
                raise HTTPException(404, "Unknown or expired streaming session")
            cache, _, offset = sessions[session_id]
            try:
                output = model(
                    features,
                    mask,
                    speaker_cache=cache,
                    num_lookahead_frames=request.num_lookahead_frames,
                    num_lookback_frames=request.num_lookback_frames,
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            frames = output.logits.shape[1]
            if request.is_last:
                del sessions[session_id]
            else:
                sessions[session_id] = (output.speaker_cache, time.monotonic(), offset + frames)
        return {
            "frame_offset": offset,
            "frame_seconds": frontend.hop_length / frontend.sampling_rate,
            "probabilities": output.probabilities[0].tolist(),
            "is_last": request.is_last,
        }

    @app.delete(
        "/v1/diarization/sessions/{session_id}",
        status_code=204,
        dependencies=[Depends(authenticate)],
    )
    def delete_session(session_id: str):
        with lock:
            expire()
            if sessions.pop(session_id, None) is None:
                raise HTTPException(404, "Unknown or expired streaming session")

    return app
