import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from nemotron_diarization_max.audio import FeatureExtractor
from nemotron_diarization_max.service import create_app


def test_http_audio_auth_and_session_lifecycle(reference_pair):
    model, _ = reference_pair
    frontend = FeatureExtractor(feature_size=4)
    app = create_app(
        model=model, frontend=frontend, api_key="test-key", max_audio_seconds=1, max_sessions=1
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-key"}
    assert client.get("/healthz").json()["device"] == "cpu"
    assert client.post("/v1/diarization/sessions").status_code == 401
    response = client.post("/v1/diarization/sessions", headers=headers)
    assert response.status_code == 201
    key = response.json()["session_id"]
    assert client.post("/v1/diarization/sessions", headers=headers).status_code == 429
    route = f"/v1/diarization/sessions/{key}"
    # Invalid input must not remove the session or advance its frame offset.
    assert (
        client.post(
            route, headers=headers, json={"input_features": [[0.0]], "is_last": True}
        ).status_code
        == 422
    )
    chunk = np.random.default_rng(0).normal(size=(8, 4)).tolist()
    response = client.post(
        route, headers=headers, json={"input_features": chunk, "num_lookahead_frames": 1}
    )
    assert response.status_code == 200
    assert response.json()["frame_offset"] == 0
    assert len(response.json()["probabilities"]) == 6
    response = client.post(
        route, headers=headers, json={"input_features": chunk[:3], "is_last": True}
    )
    assert response.status_code == 200
    assert response.json()["frame_offset"] == 6
    assert len(response.json()["probabilities"]) == 3
    assert client.delete(route, headers=headers).status_code == 404
    wav = io.BytesIO()
    sf.write(wav, np.zeros(1600, np.float32), 16000, format="WAV")
    response = client.post(
        "/v1/audio/diarizations",
        headers=headers,
        files={"file": ("silence.wav", wav.getvalue(), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["duration"] == 0.1
    assert all(s["end"] <= 0.1 for s in response.json()["segments"])
    assert (
        client.post(
            "/v1/audio/diarizations",
            headers=headers,
            files={"file": ("bad.wav", b"not an audio file")},
        ).status_code
        == 422
    )


def test_upload_and_decoded_duration_limits(reference_pair):
    model, _ = reference_pair
    client = TestClient(create_app(model=model, max_upload_bytes=64))
    assert (
        client.post("/v1/audio/diarizations", files={"file": ("big.wav", b"x" * 65)}).status_code
        == 413
    )
    client = TestClient(create_app(model=model, max_audio_seconds=0.01))
    wav = io.BytesIO()
    sf.write(wav, np.zeros(320, np.float32), 16000, format="WAV")
    assert (
        client.post(
            "/v1/audio/diarizations", files={"file": ("long.wav", wav.getvalue())}
        ).status_code
        == 413
    )
