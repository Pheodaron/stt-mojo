from concurrent.futures import ThreadPoolExecutor

import numpy as np
from starlette.testclient import TestClient

from nemotron_diarization_max.audio import FeatureExtractor
from nemotron_diarization_max.service import create_app

HEADERS = {"Content-Type": "application/octet-stream"}


def decode(response):
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["x-dtype"] == "float32-le"
    assert float(response.headers["x-frame-seconds"]) == 0.01
    shape = tuple(map(int, response.headers["x-shape"].split(",")))
    return np.frombuffer(response.content, dtype="<f4").reshape(shape)


def test_binary_features_match_direct_inference(reference_pair):
    model, _ = reference_pair
    features = np.random.default_rng(5).normal(size=(13, 4)).astype("<f4")
    expected = model(features[None]).logits[0]
    with TestClient(create_app(model=model)) as client:
        actual = decode(client.post("/infer/features", headers=HEADERS, content=features.tobytes()))
    np.testing.assert_array_equal(actual, expected)


def test_binary_audio_excludes_frontend_padding(reference_pair):
    model, _ = reference_pair
    frontend = FeatureExtractor(feature_size=4)
    audio = np.random.default_rng(6).normal(0, 0.1, 1607).astype("<f4")
    inputs = frontend(audio)
    expected = model(inputs.input_features, inputs.attention_mask).logits[0, :10]
    with TestClient(create_app(model=model, frontend=frontend)) as client:
        actual = decode(client.post("/infer/audio", headers=HEADERS, content=audio.tobytes()))
    assert actual.shape == (10, 2)
    np.testing.assert_array_equal(actual, expected)


def test_concurrent_requests_remain_independent(reference_pair):
    model, _ = reference_pair
    rng = np.random.default_rng(7)
    inputs = [rng.normal(size=(frames, 4)).astype("<f4") for frames in (19, 7, 23, 13)]
    expected = [model(features[None]).logits[0].copy() for features in inputs]
    with TestClient(create_app(model=model)) as client, ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda features: decode(
                    client.post("/infer/features", headers=HEADERS, content=features.tobytes())
                ),
                inputs,
            )
        )
    for actual, reference in zip(results, expected):
        np.testing.assert_array_equal(actual, reference)


def test_only_inference_routes_are_exposed(reference_pair):
    model, _ = reference_pair
    app = create_app(model=model)
    assert {route.path for route in app.routes} == {"/infer/audio", "/infer/features"}


def test_reject_invalid_binary_input(reference_pair):
    model, _ = reference_pair
    with TestClient(create_app(model=model)) as client:
        for route, payload in [
            ("/infer/audio", b""),
            ("/infer/audio", b"x"),
            ("/infer/audio", np.zeros(159, dtype="<f4").tobytes()),
            ("/infer/features", np.zeros(5, dtype="<f4").tobytes()),
            ("/infer/features", np.full(4, np.nan, dtype="<f4").tobytes()),
            ("/infer/audio", np.full(320, np.inf, dtype="<f4").tobytes()),
        ]:
            assert client.post(route, headers=HEADERS, content=payload).status_code == 422
        assert client.post("/infer/features", json=[[0, 0, 0, 0]]).status_code == 415
