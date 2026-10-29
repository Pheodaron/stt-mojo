"""Arrival-Order Speaker Cache policy, matching the upstream Sortformer algorithm.

State management runs on the CPU in NumPy. Encoder computation runs in MAX.
Adapted from Hugging Face Transformers Nemotron3DiarizationSpeakerCache
(Apache-2.0); see NOTICE and docs/architecture.md for source revisions.
"""

import math

import numpy as np
from scipy.special import expit

from .config import StreamingConfig


class SpeakerCache:
    def __init__(self, config: StreamingConfig, *, fifo_length=None, update_period=None):
        self.config = config
        self.fifo_length = config.fifo_length if fifo_length is None else fifo_length
        self.update_period = (
            config.speaker_cache_update_period if update_period is None else update_period
        )
        self.embeds = None
        self.probs = None
        self.fifo = None
        self.is_compressed = False

    def get_embeds(self, chunk):
        b, _, d = chunk.shape
        if self.embeds is None:
            self.embeds = np.empty((b, 0, d), np.float32)
            self.probs = np.empty((b, 0, self.config.num_speakers), np.float32)
            self.fifo = np.empty((b, 0, d), np.float32)
        if self.embeds.shape[0] != b or self.embeds.shape[2] != d:
            raise ValueError("A speaker cache belongs to a fixed batch of streams")
        return np.concatenate([self.embeds, self.fifo], axis=1)

    def update(
        self, inputs, logits, silence, num_chunk_frames, *, mask=None, num_lookback_frames=0
    ):
        self.get_embeds(inputs)
        n_cache, n_fifo = self.embeds.shape[1], self.fifo.shape[1]
        factor = self.config.subsampling_factor
        b, t, _ = inputs.shape
        probs = expit(logits).reshape(b, t, factor, self.config.num_speakers).mean(axis=2)
        if mask is not None:
            probs *= mask[..., None]
        start = n_cache + n_fifo + num_lookback_frames
        fifo = np.concatenate([self.fifo, inputs[:, start : start + num_chunk_frames]], axis=1)
        pop = (
            0
            if fifo.shape[1] <= self.fifo_length
            else min(
                max(self.update_period, fifo.shape[1] - self.fifo_length),
                fifo.shape[1],
            )
        )
        if pop:
            fifo_probs = np.concatenate(
                [probs[:, n_cache : n_cache + n_fifo], probs[:, start : start + num_chunk_frames]],
                axis=1,
            )
            stored = self.probs if self.is_compressed else probs[:, :n_cache]
            self.embeds = np.concatenate([self.embeds, fifo[:, :pop]], axis=1)
            self.probs = np.concatenate([stored, fifo_probs[:, :pop]], axis=1)
            fifo = fifo[:, pop:]
            if self.embeds.shape[1] > self.config.speaker_cache_length:
                self.embeds, self.probs = self._compress(self.embeds, self.probs, silence)
                self.is_compressed = True
        self.fifo = fifo.copy()

    @staticmethod
    def _top_indices(scores, k, axis):
        # Deterministic tie-breaking. Upstream torch.topk does not guarantee tie order.
        # Sorting descending on the requested axis keeps lower original indices on ties.
        indices = np.argsort(-scores, axis=axis, kind="stable")
        return np.take(indices, np.arange(min(k, scores.shape[axis])), axis=axis)

    def _scores(self, probs):
        c = self.config
        log_p = np.log(np.maximum(probs, c.prediction_score_threshold))
        log_complement = np.log(np.maximum(1.0 - probs, c.prediction_score_threshold))
        scores = log_p - log_complement + log_complement.sum(axis=-1, keepdims=True) - math.log(0.5)
        speech = probs > 0.5
        scores[~speech] = -np.inf
        positive = scores > 0
        budget = (
            c.speaker_cache_length // c.num_speakers - c.speaker_cache_silence_frames_per_speaker
        )
        enough = positive.sum(axis=1, keepdims=True) >= math.floor(
            budget * c.min_positive_scores_rate
        )
        scores[~positive & speech & enough] = -np.inf
        return scores

    def _compress(self, embeds, probs, silence):
        c = self.config
        b, frames, speakers = probs.shape
        scores = self._scores(probs)
        scores[:, c.speaker_cache_length :] += c.latest_frames_score_boost
        budget = c.speaker_cache_length // speakers - c.speaker_cache_silence_frames_per_speaker
        for rate, boost in [
            (c.strong_boost_rate, -2 * math.log(0.5)),
            (c.weak_boost_rate, -math.log(0.5)),
        ]:
            idx = self._top_indices(scores, math.floor(budget * rate), axis=1)
            np.put_along_axis(scores, idx, np.take_along_axis(scores, idx, axis=1) + boost, axis=1)
        scores = np.pad(
            scores,
            ((0, 0), (0, c.speaker_cache_silence_frames_per_speaker), (0, 0)),
            constant_values=np.inf,
        )
        embeds = np.concatenate(
            [embeds, np.broadcast_to(silence, (b, 1, embeds.shape[-1]))], axis=1
        )
        probs = np.pad(probs, ((0, 0), (0, 1), (0, 0)))
        scored_frames = scores.shape[1]
        sentinel = scored_frames * speakers
        flat = scores.transpose(0, 2, 1).reshape(b, -1)
        indices = self._top_indices(flat, c.speaker_cache_length, axis=1)
        indices = np.where(np.take_along_axis(flat, indices, axis=1) == -np.inf, sentinel, indices)
        indices.sort(axis=1)
        frame_indices = np.where(
            indices == sentinel, frames, np.minimum(indices % scored_frames, frames)
        )
        batches = np.arange(b)[:, None]
        return embeds[batches, frame_indices].copy(), probs[batches, frame_indices].copy()
