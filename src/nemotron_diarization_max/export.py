"""Build CPU MEF artifacts once; the Go server needs no Python interpreter."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from .audio import FeatureExtractor
from .graph import MaxGraphs
from .weights import load_checkpoint


def export_bundle(checkpoint: Path, destination: Path):
    config, weights = load_checkpoint(checkpoint)
    processor = json.loads((checkpoint / "processor_config.json").read_text())
    frontend = FeatureExtractor.from_processor_config(processor)
    if frontend.sampling_rate != 16000 or frontend.hop_length != 160:
        raise ValueError("The RPC contract requires 16 kHz audio and 10 ms frames")
    if frontend.feature_size != config.audio_config.num_mel_bins:
        raise ValueError("Frontend and model mel dimensions disagree")
    if (
        frontend.n_fft & (frontend.n_fft - 1)
        or not 2 * frontend.hop_length <= frontend.n_fft <= 8192
    ):
        raise ValueError("Unsupported FFT size for the Go frontend")
    destination.mkdir(parents=True, exist_ok=True)
    # A failed rebuild must not leave a manifest advertising a mixed bundle.
    (destination / "manifest.json").unlink(missing_ok=True)
    graphs = MaxGraphs(config, weights, export_dir=destination)
    save_file(
        {**weights, "__upsample_kernel": graphs._upsample_kernel},
        destination / "weights.safetensors",
    )
    a, s = config.audio_config, config.streaming_config
    dim = int(
        a.hidden_size // a.num_attention_heads * a.rope_parameters.get("partial_rotary_factor", 1.0)
    )
    inv = 1 / (
        np.float32(a.rope_parameters.get("rope_theta", 10000.0))
        ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    )
    # Export float32 constants verbatim to avoid different mel/RoPE rounding in Go.
    constants = np.concatenate(
        [
            frontend.window.ravel(),
            frontend.mel_filters.ravel(),
            inv,
            weights["silence_embeds"].ravel(),
        ]
    )
    constants.astype("<f4").tofile(destination / "constants.f32")
    manifest = dict(
        version=1,
        max_version="26.7.0",
        target="cpu",
        sample_rate=16000,
        hop_length=frontend.hop_length,
        n_fft=frontend.n_fft,
        preemphasis=frontend.preemphasis,
        mel_bins=a.num_mel_bins,
        hidden_size=a.hidden_size,
        subsampling_factor=a.subsampling_factor,
        rotary_dim=dim,
        num_speakers=config.head_config.num_speakers,
        chunk_length=config.chunk_length,
        right_context=config.chunk_right_context,
        left_context=config.chunk_left_context,
        fifo_length=config.fifo_length,
        update_period=config.speaker_cache_update_period,
        cache_length=s.speaker_cache_length,
        silence_frames=s.speaker_cache_silence_frames_per_speaker,
        prediction_threshold=s.prediction_score_threshold,
        latest_boost=s.latest_frames_score_boost,
        strong_boost=s.strong_boost_rate,
        weak_boost=s.weak_boost_rate,
        min_positive_rate=s.min_positive_scores_rate,
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_bundle(args.model, args.output)


if __name__ == "__main__":
    main()
