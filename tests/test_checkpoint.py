import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from nemotron_diarization_max import ModelConfig
from nemotron_diarization_max.weights import expected_shapes, load_checkpoint


def test_single_and_sharded_checkpoint_roundtrip(tmp_path):
    config = ModelConfig.from_dict(
        {
            "audio_config": {
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "num_mel_bins": 128,
                "subsampling_factor": 2,
            },
            "head_config": {
                "audio_hidden_size": 256,
                "hidden_size": 64,
                "num_speakers": 2,
                "subsampling_factor": 2,
            },
            "streaming_config": {"num_speakers": 2, "subsampling_factor": 2},
        }
    )
    source, output = tmp_path / "source", tmp_path / "sharded"
    source.mkdir()
    weights = {
        name: np.full(shape, i / 100, np.float32)
        for i, (name, shape) in enumerate(expected_shapes(config).items())
    }
    (source / "config.json").write_text(json.dumps(config.to_dict()))
    save_file(weights, source / "model.safetensors")
    actual_config, actual = load_checkpoint(source)
    assert actual_config == config
    script = Path(__file__).parents[1] / "scripts/shard_checkpoint.py"
    subprocess.run(
        [sys.executable, str(script), str(source), str(output), "--max-mib", "1"], check=True
    )
    assert len(list(output.glob("*.safetensors"))) > 1
    actual_config, actual = load_checkpoint(output)
    assert actual_config == config
    for key in weights:
        np.testing.assert_array_equal(actual[key], weights[key])

    # Hugging Face snapshot files are symlinks to a sibling blobs directory.
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").symlink_to(source / "config.json")
    (snapshot / "model.safetensors").symlink_to(source / "model.safetensors")
    _, linked = load_checkpoint(snapshot)
    for key in weights:
        np.testing.assert_array_equal(linked[key], weights[key])


def test_reject_shard_path_outside_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(ModelConfig().to_dict()))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../escape.safetensors"}})
    )
    with pytest.raises(ValueError, match="escapes"):
        load_checkpoint(tmp_path)
