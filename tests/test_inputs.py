import numpy as np
import pytest

from nemotron_diarization_max import ModelConfig
from nemotron_diarization_max.audio import FeatureExtractor, speaker_segments
from nemotron_diarization_max.weights import validate_weights


def test_overlap_segments_and_padding():
    probabilities = np.array([[0.8, 0.1], [0.8, 0.8], [0.1, 0.8], [0.9, 0.9]])
    assert speaker_segments(probabilities, attention_mask=[1, 1, 1, 0]) == [
        {"speaker": 0, "start": 0.0, "end": 0.02},
        {"speaker": 1, "start": 0.01, "end": 0.03},
    ]


def test_reject_incompatible_checkpoint():
    with pytest.raises(ValueError, match="Expected"):
        ModelConfig.from_dict({"model_type": "llama"})
    with pytest.raises(ValueError, match="RoPE"):
        ModelConfig.from_dict({"audio_config": {"rope_parameters": {"rope_type": "dynamic"}}})
    with pytest.raises(ValueError, match="Head and cache"):
        ModelConfig.from_dict({"head_config": {"num_speakers": 4}})
    with pytest.raises(ValueError, match="missing"):
        validate_weights(ModelConfig(), {})


def test_reject_invalid_audio():
    frontend = FeatureExtractor()
    for audio in [np.array([]), np.array([np.nan] * 320), np.zeros((320, 2))]:
        with pytest.raises(ValueError):
            frontend(audio)
    with pytest.raises(ValueError, match="16000"):
        frontend(np.zeros(320), sampling_rate=8000)


def test_reject_invalid_context_and_mask(reference_pair):
    model, _ = reference_pair
    with pytest.raises(ValueError, match="leave at least"):
        model(np.zeros((1, 8, 4), np.float32), num_lookahead_frames=4)
    with pytest.raises(ValueError, match="Attention mask"):
        model(np.zeros((1, 8, 4), np.float32), np.zeros((1, 8), bool))
