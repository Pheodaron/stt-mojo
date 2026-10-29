import os

import pytest

# The managed environment's default ~/.cache is read-only. Set before MAX imports.
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nemotron-max-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/nemotron-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nemotron-mpl-cache")


@pytest.fixture(scope="session")
def reference_pair():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Nemotron3DiarizationConfig"):
        pytest.skip("Install the pinned [reference] extra")
    from nemotron_diarization_max import ModelConfig, NemotronDiarization

    torch.set_num_threads(2)
    config = transformers.Nemotron3DiarizationConfig(
        audio_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_mel_bins": 4,
            "subsampling_factor": 2,
        },
        head_config={
            "hidden_size": 8,
            "audio_hidden_size": 16,
            "num_speakers": 2,
            "subsampling_factor": 2,
        },
        streaming_config={
            "num_speakers": 2,
            "subsampling_factor": 2,
            "speaker_cache_length": 8,
            "fifo_length": 3,
            "speaker_cache_update_period": 2,
        },
        chunk_length=3,
        chunk_right_context=1,
        chunk_left_context=1,
        fifo_length=3,
        speaker_cache_update_period=2,
    )
    torch.manual_seed(321)
    reference = transformers.Nemotron3DiarizationForAudioFrameClassification(config).eval()
    reference.config.audio_config._attn_implementation = "eager"
    # Nontrivial biases and all three convolution taps exercise more than identity initialization.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "layer_norm" in name and name.endswith("weight"):
                parameter.normal_(1.0, 0.04)
            elif parameter.ndim >= 2:
                parameter.normal_(0.0, 0.5 / parameter.shape[1] ** 0.5)
            else:
                parameter.normal_(0.0, 0.04)
    weights = {k: v.detach().numpy().copy() for k, v in reference.state_dict().items()}
    model = NemotronDiarization(ModelConfig.from_dict(config.to_dict()), weights)
    return model, reference
