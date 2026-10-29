import numpy as np
import pytest

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("batch,frames", [(1, 1), (1, 13), (2, 19), (1, 51)])
def test_offline_logits_match_reference(reference_pair, batch, frames):
    import torch

    model, reference = reference_pair
    features = np.random.default_rng(frames).normal(size=(batch, frames, 4)).astype(np.float32)
    mask = np.ones((batch, frames), bool)
    if batch > 1:
        mask[1, -4:] = False
        features[1, -4:] = 0
    actual = model(features, mask)
    with torch.no_grad():
        expected = reference(
            torch.from_numpy(features), attention_mask=torch.from_numpy(mask)
        ).logits.numpy()
    np.testing.assert_allclose(actual.logits, expected, atol=2e-5, rtol=2e-5)
    assert actual.speaker_cache is None
    assert actual.logits.shape == (batch, frames, 2)


def test_streaming_cache_compression_and_contexts(reference_pair):
    import torch

    model, reference = reference_pair
    cache = model.new_cache()
    reference_cache = None
    rng = np.random.default_rng(123)
    for step in range(12):
        last = step == 11
        features = rng.normal(size=(1, 3 if last else 8, 4)).astype(np.float32)
        left, right = (0, 0) if last else (1, 1)
        actual = model(
            features, speaker_cache=cache, num_lookahead_frames=right, num_lookback_frames=left
        )
        with torch.no_grad():
            expected = reference(
                torch.from_numpy(features),
                speaker_cache=reference_cache,
                num_lookahead_frames=right,
                num_lookback_frames=left,
            )
        np.testing.assert_allclose(actual.logits, expected.logits.numpy(), atol=3e-5, rtol=3e-5)
        cache, reference_cache = actual.speaker_cache, expected.speaker_cache
        np.testing.assert_allclose(
            cache.embeds,
            reference_cache.embeds[:, : reference_cache.num_cache_frames].numpy(),
            atol=3e-5,
            rtol=3e-5,
        )
        np.testing.assert_allclose(
            cache.fifo,
            reference_cache.fifo[:, : reference_cache.num_fifo_frames].numpy(),
            atol=3e-5,
            rtol=3e-5,
        )
        np.testing.assert_allclose(
            cache.probs,
            reference_cache.probs[:, : reference_cache.num_cache_frames].numpy(),
            atol=3e-5,
            rtol=3e-5,
        )
    assert cache.is_compressed and reference_cache.is_compressed


def test_independent_recordings_do_not_share_state(reference_pair):
    model, _ = reference_pair
    rng = np.random.default_rng(7)
    features = rng.normal(size=(1, 27, 4)).astype(np.float32)
    expected = model(features).logits.copy()
    model(rng.normal(size=(1, 15, 4)).astype(np.float32))
    np.testing.assert_array_equal(model(features).logits, expected)


def test_feature_frontend_matches_upstream():
    from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (
        NemotronAsrStreamingFeatureExtractor,
    )

    from nemotron_diarization_max.audio import FeatureExtractor

    frontend = FeatureExtractor()
    reference = NemotronAsrStreamingFeatureExtractor(feature_size=128)
    rng = np.random.default_rng(10)
    for center, last, count in [
        (True, False, 16000),
        (False, False, 8192),
        (False, True, 8350),
        (True, False, 160),
    ]:
        audio = rng.normal(0, 0.1, count).astype(np.float32)
        actual = frontend(audio, center=center, is_last_audio_chunk=last)
        expected = reference(
            audio, sampling_rate=16000, center=center, is_last_audio_chunk=last, return_tensors="np"
        )
        np.testing.assert_allclose(
            actual.input_features, expected.input_features, atol=3e-5, rtol=3e-5
        )
        np.testing.assert_array_equal(actual.attention_mask, expected.attention_mask)


def test_cache_selection_matches_upstream():
    import torch
    from transformers import Nemotron3DiarizationStreamingConfig
    from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
        Nemotron3DiarizationSpeakerCache,
    )

    from nemotron_diarization_max.cache import SpeakerCache
    from nemotron_diarization_max.config import StreamingConfig

    config = StreamingConfig()
    cache = SpeakerCache(config)
    reference = Nemotron3DiarizationSpeakerCache(Nemotron3DiarizationStreamingConfig())
    rng = np.random.default_rng(83)
    embeds = rng.normal(size=(2, 564, 5)).astype(np.float32)
    probs = rng.uniform(0.01, 0.15, size=(2, 564, 8)).astype(np.float32)
    active = rng.integers(0, 8, size=(2, 564))
    probs[np.arange(2)[:, None], np.arange(564)[None], active] = rng.uniform(
        0.6, 0.95, size=(2, 564)
    )
    silence = rng.normal(size=(5,)).astype(np.float32)
    actual = cache._compress(embeds, probs, silence)
    expected = reference._compress(
        torch.from_numpy(embeds), torch.from_numpy(probs), torch.from_numpy(silence)
    )
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e.numpy())
