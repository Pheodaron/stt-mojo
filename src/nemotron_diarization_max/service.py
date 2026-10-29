"""Stateless binary inference endpoints for the compiled CPU model."""

import json
from pathlib import Path

import numpy as np
from anyio import CapacityLimiter, to_thread
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from .audio import FeatureExtractor
from .model import NemotronDiarization


def create_app(model_path=None, *, model=None, frontend=None):
    if model is None:
        path = Path(model_path)
        if not path.is_dir():
            raise ValueError("--model must be a local checkpoint directory")
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

    # One CPU inference at a time. Wait asynchronously instead of blocking
    # request threads on a mutex; MAX uses its own CPU execution threads.
    inference = CapacityLimiter(1)
    mel_bins = model.config.audio_config.num_mel_bins

    def infer(payload: bytes, audio: bool):
        width = 1 if audio else mel_bins
        if not payload or len(payload) % (4 * width):
            raise ValueError(f"Expected nonempty little-endian float32 data, width={width}")
        values = np.frombuffer(payload, dtype="<f4")
        if audio:
            features = frontend(values, sampling_rate=frontend.sampling_rate)
            logits = model(features.input_features, features.attention_mask).logits[0]
            logits = logits[: int(features.attention_mask.sum())]
        else:
            logits = model(values.reshape(1, -1, mel_bins)).logits[0]
        return Response(
            logits.astype("<f4", copy=False).tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Shape": f"{logits.shape[0]},{logits.shape[1]}",
                "X-Dtype": "float32-le",
                "X-Frame-Seconds": str(frontend.hop_length / frontend.sampling_rate),
            },
        )

    async def dispatch(request: Request, audio: bool):
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip()
            != "application/octet-stream"
        ):
            raise HTTPException(415, "Use application/octet-stream with little-endian float32 data")
        payload = await request.body()
        try:
            return await to_thread.run_sync(infer, payload, audio, limiter=inference)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    async def audio(request: Request):
        return await dispatch(request, audio=True)

    async def features(request: Request):
        return await dispatch(request, audio=False)

    return Starlette(
        routes=[
            Route("/infer/audio", audio, methods=["POST"]),
            Route("/infer/features", features, methods=["POST"]),
        ]
    )
