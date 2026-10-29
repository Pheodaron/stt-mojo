from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit

from .cache import SpeakerCache
from .config import ModelConfig
from .weights import load_checkpoint, resolve_checkpoint, validate_weights


@dataclass
class DiarizationOutput:
    logits: np.ndarray
    speaker_cache: SpeakerCache | None = None

    @property
    def probabilities(self):
        return expit(self.logits)


class NemotronDiarization:
    """CPU/float32 model with explicit, per-session streaming state.

    Inputs are log-mel features [batch, time, mel_bins], not raw audio.
    Offline calls create isolated state. Streaming calls receive and return state.
    Do not call the same model concurrently without external serialization.
    """

    def __init__(self, config: ModelConfig, weights: dict[str, np.ndarray]):
        from .graph import MaxGraphs

        self.config = config
        self.weights = validate_weights(config, weights)
        self.graphs = MaxGraphs(config, self.weights)

    @classmethod
    def from_pretrained(cls, model: str | Path, *, revision: str | None = None):
        config, weights = load_checkpoint(resolve_checkpoint(model, revision))
        return cls(config, weights)

    def new_cache(self, *, offline=False):
        return SpeakerCache(
            self.config.streaming_config,
            **(
                {
                    "fifo_length": self.config.fifo_length,
                    "update_period": self.config.speaker_cache_update_period,
                }
                if offline
                else {}
            ),
        )

    def __call__(
        self,
        input_features,
        attention_mask=None,
        *,
        speaker_cache=None,
        num_lookahead_frames=None,
        num_lookback_frames=None,
    ):
        features = np.asarray(input_features, dtype=np.float32)
        a = self.config.audio_config
        if (
            features.ndim != 3
            or features.shape[-1] != a.num_mel_bins
            or min(features.shape[:2]) < 1
        ):
            raise ValueError(f"Expected nonempty [batch, time, {a.num_mel_bins}] features")
        if not np.isfinite(features).all():
            raise ValueError("Features must be finite")
        if attention_mask is not None:
            attention_mask = np.asarray(attention_mask, dtype=bool)
            if attention_mask.shape != features.shape[:2] or not attention_mask.any(axis=1).all():
                raise ValueError("Attention mask must match input and contain valid frames")
        streaming = (
            speaker_cache is not None
            or num_lookahead_frames is not None
            or num_lookback_frames is not None
        )
        if speaker_cache is not None and speaker_cache.config != self.config.streaming_config:
            raise ValueError("Speaker cache configuration disagrees with this model")
        cache = (
            speaker_cache if speaker_cache is not None else self.new_cache(offline=not streaming)
        )
        right = 0 if num_lookahead_frames is None else num_lookahead_frames
        left = 0 if num_lookback_frames is None else num_lookback_frames
        if type(right) is not int or type(left) is not int:
            raise ValueError("Look-ahead and look-back must be integer encoder frame counts")
        n_embeds = (features.shape[1] + a.subsampling_factor - 1) // a.subsampling_factor
        if left < 0 or right < 0 or n_embeds - right - left < 1:
            raise ValueError("Contexts must be nonnegative and leave at least one frame to score")
        embeds = self.graphs.embed(features)
        end = embeds.shape[1] - right
        factor = a.subsampling_factor
        mask = None if attention_mask is None else attention_mask[:, ::factor]
        step = end - left if streaming else self.config.chunk_length
        lc = left if streaming else self.config.chunk_left_context
        rc = right if streaming else self.config.chunk_right_context
        logits = []
        for start in range(left, end, step):
            stop = min(start + step, end)
            context_start = max(0, start - lc)
            chunk = embeds[:, context_start : min(stop + rc, embeds.shape[1])]
            cached = cache.get_embeds(chunk)
            inputs = np.concatenate([cached, chunk], axis=1)
            valid = (
                None
                if mask is None
                else np.concatenate(
                    [
                        np.ones((features.shape[0], cached.shape[1]), bool),
                        mask[:, context_start : context_start + chunk.shape[1]],
                    ],
                    axis=1,
                )
            )
            prediction = self.graphs.infer(inputs, valid)
            lookback = start - context_start
            cache.update(
                inputs,
                prediction,
                self.weights["silence_embeds"],
                stop - start,
                mask=valid,
                num_lookback_frames=lookback,
            )
            lo = (cached.shape[1] + lookback) * factor
            logits.append(prediction[:, lo : lo + (stop - start) * factor])
        return DiarizationOutput(
            np.concatenate(logits, axis=1)[:, : features.shape[1] - left * factor],
            cache if streaming else None,
        )
