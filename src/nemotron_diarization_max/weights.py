import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from .config import ModelConfig


def expected_shapes(c: ModelConfig) -> dict[str, tuple[int, ...]]:
    a, h = c.audio_config, c.head_config
    d, m, p, f = a.hidden_size, a.intermediate_size, h.hidden_size, a.subsampling_factor
    shapes = {
        "silence_embeds": (d,),
        "model.audio_tower.embedder.projection.weight": (d, f * a.num_mel_bins),
    }

    def linear(name, out_size, in_size, bias=True):
        shapes[name + ".weight"] = (out_size, in_size)
        if bias:
            shapes[name + ".bias"] = (out_size,)

    def norm(name):
        shapes[name + ".weight"] = (d,)
        shapes[name + ".bias"] = (d,)

    norm("model.audio_tower.input_layer_norm")
    norm("model.audio_tower.layer_norm")
    for i in range(a.num_hidden_layers):
        base = f"model.audio_tower.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj"):
            linear(base + ".self_attn." + proj, d, d, False)
        linear(base + ".self_attn.o_proj", d, d)
        norm(base + ".layer_norm1")
        norm(base + ".layer_norm2")
        linear(base + ".mlp.fc1", m, d)
        linear(base + ".mlp.fc2", d, m)
    linear("model.proj", p, d)
    shapes["model.upsampler.conv.weight"] = (p * f, p, 3)
    shapes["model.upsampler.conv.bias"] = (p * f,)
    linear("classifier.dense", p, p)
    linear("classifier.out_proj", h.num_speakers, p)
    return shapes


def validate_weights(c: ModelConfig, weights: dict) -> dict[str, np.ndarray]:
    shapes = expected_shapes(c)
    missing, extra = shapes.keys() - weights.keys(), weights.keys() - shapes.keys()
    if missing or extra:
        raise ValueError(
            f"Checkpoint keys disagree: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result = {}
    for name, shape in shapes.items():
        value = np.asarray(weights[name])
        if value.shape != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite weights")
        result[name] = np.ascontiguousarray(value, dtype=np.float32)
    return result


def resolve_checkpoint(model: str | Path, revision: str | None = None) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    if path.exists() or str(model).endswith(".nemo"):
        raise ValueError(
            "Use a Transformers safetensors directory; raw .nemo archives are not supported"
        )
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(model),
            revision=revision,
            allow_patterns=[
                "config.json",
                "processor_config.json",
                "*.safetensors",
                "*.safetensors.index.json",
            ],
        )
    )


def load_checkpoint(path: Path):
    config = ModelConfig.load(path / "config.json")
    index = path / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    weights = {}
    for filename in files:
        # HF snapshots legitimately link each file to a sibling blobs directory.
        # Validate the index's filename without rejecting those cache symlinks.
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".safetensors")
        ):
            raise ValueError("Safetensors shard escapes the checkpoint directory")
        shard = load_file(path / filename)
        if weights.keys() & shard.keys():
            raise ValueError("Duplicate tensor names across checkpoint shards")
        weights.update(shard)
    return config, validate_weights(config, weights)
