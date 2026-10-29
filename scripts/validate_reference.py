"""Compare actual MAX executions against the pinned Transformers implementation.

Run with --model for trained weights, or --synthetic to validate the full topology
with deterministic random weights. These are distinct validation claims.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Nemotron3DiarizationConfig, Nemotron3DiarizationForAudioFrameClassification

from nemotron_diarization_max import ModelConfig, NemotronDiarization
from nemotron_diarization_max.weights import resolve_checkpoint


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Local trained safetensors directory")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--config", help="Optional config.json for synthetic validation")
    parser.add_argument("--output", default="validation-local.json")
    parser.add_argument("--atol", type=float, default=5e-5)
    parser.add_argument("--stream-steps", type=int, default=4)
    args = parser.parse_args()
    if args.stream_steps < 2:
        parser.error("--stream-steps must be at least 2")
    torch.set_num_threads(2)
    torch.manual_seed(20261029)
    config = ModelConfig.load(args.config) if args.config else ModelConfig()
    if args.synthetic:
        reference = Nemotron3DiarizationForAudioFrameClassification(
            Nemotron3DiarizationConfig(**config.to_dict())
        ).eval()
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if "layer_norm" in name and name.endswith("weight"):
                    parameter.normal_(1, 0.04)
                elif parameter.ndim >= 2:
                    parameter.normal_(0, 0.5 / parameter.shape[1] ** 0.5)
                else:
                    parameter.normal_(0, 0.04)
        config = ModelConfig.from_dict(reference.config.to_dict())
    else:
        path = resolve_checkpoint(args.model)
        reference = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(
            path, local_files_only=True, dtype=torch.float32
        ).eval()
        config = ModelConfig.load(path / "config.json")
    reference.config.audio_config._attn_implementation = "eager"
    weights = {k: v.detach().numpy() for k, v in reference.state_dict().items()}
    started = time.perf_counter()
    print("Compiling MAX graphs on CPU...", flush=True)
    model = NemotronDiarization(config, weights)
    report = {
        "weights": "synthetic" if args.synthetic else "trained",
        "device": "cpu",
        "dtype": "float32",
        "layers": config.audio_config.num_hidden_layers,
        "compile_seconds": time.perf_counter() - started,
        "checks": [],
    }

    def compare(name, actual, expected, elapsed):
        error = np.abs(actual - expected)
        passed = bool(np.allclose(actual, expected, atol=args.atol, rtol=args.atol))
        row = {
            "case": name,
            "shape": list(actual.shape),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
            "passed": passed,
            "max_seconds": elapsed,
        }
        report["checks"].append(row)
        if not passed or name == "offline_masked" or len(report["checks"]) % 10 == 0:
            print(json.dumps(row), flush=True)
        return passed

    rng = np.random.default_rng(20261029)
    mel = config.audio_config.num_mel_bins
    features = rng.normal(0, 1, (1, 137, mel)).astype(np.float32)
    mask = np.ones(features.shape[:2], bool)
    mask[:, -5:] = False
    features[:, -5:] = 0
    started = time.perf_counter()
    actual = model(features, mask).logits
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        expected = reference(
            torch.from_numpy(features), attention_mask=torch.from_numpy(mask)
        ).logits.numpy()
    compare("offline_masked", actual, expected, elapsed)
    cache, reference_cache = model.new_cache(), None
    factor = config.audio_config.subsampling_factor
    for step in range(args.stream_steps):
        last = step == args.stream_steps - 1
        features = rng.normal(0, 1, (1, 17 if last else 13 * factor, mel)).astype(np.float32)
        right = 0 if last else 4
        started = time.perf_counter()
        actual = model(features, speaker_cache=cache, num_lookahead_frames=right)
        elapsed = time.perf_counter() - started
        with torch.no_grad():
            expected = reference(
                torch.from_numpy(features),
                speaker_cache=reference_cache,
                num_lookahead_frames=right,
            )
        compare(f"stream_{step}", actual.logits, expected.logits.numpy(), elapsed)
        cache, reference_cache = actual.speaker_cache, expected.speaker_cache
    report["stream_cache_compressed"] = cache.is_compressed
    report["stream_cache_frames"] = cache.embeds.shape[1]
    report["stream_fifo_frames"] = cache.fifo.shape[1]
    report["passed"] = all(row["passed"] for row in report["checks"])
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
