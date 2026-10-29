"""NumPy/SciPy implementation of the checkpoint's unnormalized log-mel frontend."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft
from scipy.signal import resample_poly


def _mel_filterbank(sample_rate, n_fft, n_mels):
    # Slaney scale and area normalization, including librosa's float32 rounding.
    def hz_to_mel(hz):
        hz = np.asarray(hz, dtype=np.float64)
        return np.where(
            hz >= 1000,
            15 + np.log(np.maximum(hz, 1e-30) / 1000) / (np.log(6.4) / 27),
            hz / (200 / 3),
        )

    points = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)
    hz = np.where(points >= 15, 1000 * np.exp((points - 15) * np.log(6.4) / 27), points * (200 / 3))
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    ramps = hz[:, None] - frequencies[None]
    diffs = np.diff(hz)
    weights = np.empty((n_mels, len(frequencies)), np.float32)
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / diffs[i], ramps[i + 2] / diffs[i + 1]))
    weights *= (2.0 / (hz[2:] - hz[:-2]))[:, None]
    return weights


@dataclass
class AudioFeatures:
    input_features: np.ndarray
    attention_mask: np.ndarray


class FeatureExtractor:
    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=512,
        win_length=400,
        preemphasis=0.97,
    ):
        if sampling_rate <= 0 or hop_length <= 0 or not 0 < win_length <= n_fft or feature_size < 1:
            raise ValueError("Invalid audio frontend configuration")
        self.feature_size = feature_size
        self.sampling_rate, self.hop_length = sampling_rate, hop_length
        self.n_fft, self.win_length, self.preemphasis = n_fft, win_length, preemphasis
        window = np.hanning(win_length).astype(np.float32)
        padding = n_fft - win_length
        self.window = np.pad(window, (padding // 2, padding - padding // 2))
        self.mel_filters = _mel_filterbank(sampling_rate, n_fft, feature_size)

    @classmethod
    def from_processor_config(cls, config: dict):
        value = config.get("feature_extractor", {})
        if (
            value.get("feature_extractor_type", "NemotronAsrStreamingFeatureExtractor")
            != "NemotronAsrStreamingFeatureExtractor"
        ):
            raise ValueError("Unsupported audio frontend")
        if value.get("mel_filters_dtype", "float32") != "float32":
            raise ValueError("Only the float32 diarization frontend is supported")
        keys = ("feature_size", "sampling_rate", "hop_length", "n_fft", "win_length", "preemphasis")
        return cls(**{k: value[k] for k in keys if k in value})

    def __call__(self, audio, *, sampling_rate=16000, center=True, is_last_audio_chunk=False):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Audio must be a finite mono waveform")
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Expected {self.sampling_rate} Hz audio")
        if audio.size < self.hop_length:
            raise ValueError("Audio must contain at least one 10 ms frame")
        waveform = audio.copy()
        if self.preemphasis is not None:
            waveform[1:] = audio[1:] - np.float32(self.preemphasis) * audio[:-1]
        if is_last_audio_chunk and not center:
            waveform = np.pad(waveform, (0, self.n_fft // 2 - self.hop_length))
        if center:
            valid_length = len(waveform) // self.hop_length
            waveform = np.pad(waveform, (self.n_fft // 2, self.n_fft // 2))
        else:
            valid_length = (len(waveform) - self.n_fft) // self.hop_length + 1
        if len(waveform) < self.n_fft:
            raise ValueError("Uncentered chunk is shorter than one FFT window")
        frames = np.lib.stride_tricks.sliding_window_view(waveform, self.n_fft)[:: self.hop_length]
        spectrum = rfft(frames * self.window, axis=-1)
        # Keep the magnitude -> square order used by the upstream frontend.
        magnitude = np.sqrt(spectrum.real**2 + spectrum.imag**2)
        mel = (magnitude**2) @ self.mel_filters.T
        features = np.log(mel + np.float32(2**-24)).astype(np.float32)
        mask = np.arange(len(features)) < valid_length
        features *= mask[:, None]
        return AudioFeatures(features[None], mask[None])


def resample_mono(audio, sample_rate, target_rate=16000):
    audio = np.asarray(audio, np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or sample_rate <= 0:
        raise ValueError("Invalid audio shape or sample rate")
    if sample_rate != target_rate:
        gcd = math.gcd(sample_rate, target_rate)
        audio = resample_poly(audio, target_rate // gcd, sample_rate // gcd)
    return np.ascontiguousarray(audio, np.float32)


def speaker_segments(
    probabilities, *, threshold=0.5, frame_seconds=0.01, attention_mask=None, duration=None
):
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or not 0 <= threshold <= 1:
        raise ValueError("Expected [time, speakers] probabilities and threshold in [0,1]")
    active = probabilities > threshold
    if attention_mask is not None:
        active &= np.asarray(attention_mask, bool)[:, None]
    changes = np.diff(np.pad(active.astype(np.int8), ((1, 1), (0, 0))), axis=0)
    segments = []
    for speaker in range(active.shape[1]):
        starts, ends = (
            np.flatnonzero(changes[:, speaker] == 1),
            np.flatnonzero(changes[:, speaker] == -1),
        )
        for start, end in zip(starts, ends):
            lo, hi = float(start * frame_seconds), float(end * frame_seconds)
            if duration is not None:
                lo, hi = min(lo, duration), min(hi, duration)
            if hi > lo:
                segments.append(
                    {"speaker": int(speaker), "start": round(lo, 6), "end": round(hi, 6)}
                )
    return sorted(segments, key=lambda x: (x["start"], x["speaker"]))
