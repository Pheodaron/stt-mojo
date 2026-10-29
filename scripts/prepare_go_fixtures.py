"""Export a small random-weight model and upstream numerical fixtures for Go tests.

Run with the pinned [reference] dependencies. These are NOT pretrained weights.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import (
    Nemotron3DiarizationConfig,
    Nemotron3DiarizationForAudioFrameClassification,
)
from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (
    NemotronAsrStreamingFeatureExtractor,
)

from nemotron_diarization_max.audio import FeatureExtractor
from nemotron_diarization_max.cache import SpeakerCache
from nemotron_diarization_max.config import StreamingConfig
from nemotron_diarization_max.export import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nemo-source", type=Path, help="Pinned NeMo sortformer_modules.py")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(321)
    config = Nemotron3DiarizationConfig(
        audio_config=dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_mel_bins=4,
            subsampling_factor=2,
        ),
        head_config=dict(hidden_size=8, audio_hidden_size=16, num_speakers=2, subsampling_factor=2),
        streaming_config=dict(
            num_speakers=2,
            subsampling_factor=2,
            speaker_cache_length=8,
            fifo_length=3,
            speaker_cache_update_period=2,
        ),
        chunk_length=3,
        chunk_right_context=1,
        chunk_left_context=1,
        fifo_length=3,
        speaker_cache_update_period=2,
    )
    model = Nemotron3DiarizationForAudioFrameClassification(config).eval()
    model.config.audio_config._attn_implementation = "eager"
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "layer_norm" in name and name.endswith("weight"):
                p.normal_(1, 0.04)
            elif p.ndim >= 2:
                p.normal_(0, 0.5 / p.shape[1] ** 0.5)
            else:
                p.normal_(0, 0.04)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp)
        (checkpoint / "config.json").write_text(json.dumps(config.to_dict()))
        (checkpoint / "processor_config.json").write_text(
            json.dumps(
                {
                    "feature_extractor": {"feature_size": 4},
                }
            )
        )
        save_file(
            {k: v.numpy().copy() for k, v in model.state_dict().items()},
            checkpoint / "model.safetensors",
        )
        export_bundle(checkpoint, args.output)

    frontend = NemotronAsrStreamingFeatureExtractor(feature_size=4)
    rng = np.random.default_rng(42)
    lengths = [160, 161, 319, 320, 511, 512, 959, 960, 1120, 1280, 1600, 1607, 12017]
    for count in lengths:
        audio = rng.normal(0, 0.1, count).astype(np.float32)
        features = frontend(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            expected = model(features.input_features, attention_mask=features.attention_mask)
        audio.astype("<f4").tofile(args.output / f"audio-{count}.f32")
        features.input_features.numpy().astype("<f4").tofile(args.output / f"mel-{count}.f32")
        expected.logits[0, : count // 160].numpy().astype("<f4").tofile(
            args.output / f"logits-{count}.f32"
        )
    (args.output / "lengths.json").write_text(json.dumps(lengths))

    # The production frontend has 128 mel channels, independently of the small
    # model used above. Cover sparse filters, speech-scale samples and silence.
    audio = rng.normal(0, 0.001, 32707).astype(np.float32)
    audio += (0.1 * np.sin(np.arange(len(audio)) * 0.17)).astype(np.float32)
    audio[:1600] = 0
    audio[-1600:] = 0
    frontend128 = NemotronAsrStreamingFeatureExtractor(feature_size=128)
    expected = frontend128(audio, sampling_rate=16000, return_tensors="np")
    audio.astype("<f4").tofile(args.output / "frontend128-audio.f32")
    expected.input_features.astype("<f4").tofile(args.output / "frontend128-mel.f32")
    frontend128_constants = FeatureExtractor()
    np.concatenate(
        [frontend128_constants.window, frontend128_constants.mel_filters.ravel()]
    ).astype("<f4").tofile(args.output / "frontend128-constants.f32")

    # Policy fixture with distinct scores, eight speakers, silence and disabled
    # entries; independent of encoder numerical error and topk tie conventions.
    cache = SpeakerCache(StreamingConfig())
    embeds = np.arange(564, dtype=np.float32)[None, :, None]
    probs = rng.uniform(0.01, 0.15, (1, 564, 8)).astype(np.float32)
    active = rng.integers(0, 8, 564)
    probs[0, np.arange(564), active] = rng.uniform(0.6, 0.95, 564)
    selected, _ = cache._compress(embeds, probs, np.array([-1], np.float32))
    if args.nemo_source:
        from nemo_reference import load_cache, update_fixtures

        nemo = load_cache(args.nemo_source, speakers=8, hidden=1, capacity=264, fifo=40, update=300)
        nemo_selected, _, _ = nemo._compress_spkcache(
            torch.from_numpy(embeds), torch.from_numpy(probs), torch.zeros(1, 1)
        )
        np.testing.assert_array_equal(selected, nemo_selected.numpy())
        (args.output / "nemo-updates.json").write_text(
            json.dumps(update_fixtures(args.nemo_source, rng))
        )
    probs.astype("<f4").tofile(args.output / "cache-probs.f32")
    selected.astype("<f4").tofile(args.output / "cache-selected.f32")
    print(f"Exported random-weight CPU fixture to {args.output}")


if __name__ == "__main__":
    main()
