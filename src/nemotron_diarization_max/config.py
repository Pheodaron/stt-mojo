import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def _read(cls, value):
    if isinstance(value, cls):
        return value
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (value or {}).items() if k in known})


@dataclass(frozen=True)
class AudioConfig:
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 31
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    num_mel_bins: int = 128
    subsampling_factor: int = 8
    hidden_act: str = "gelu"
    rope_parameters: dict = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        }
    )


@dataclass(frozen=True)
class HeadConfig:
    hidden_size: int = 192
    audio_hidden_size: int = 512
    num_speakers: int = 8
    subsampling_factor: int = 8


@dataclass(frozen=True)
class StreamingConfig:
    fifo_length: int = 264
    speaker_cache_update_period: int = 222
    speaker_cache_length: int = 264
    speaker_cache_silence_frames_per_speaker: int = 1
    prediction_score_threshold: float = 0.25
    latest_frames_score_boost: float = 0.05
    strong_boost_rate: float = 0.75
    weak_boost_rate: float = 1.5
    min_positive_scores_rate: float = 0.5
    num_speakers: int = 8
    subsampling_factor: int = 8


@dataclass(frozen=True)
class ModelConfig:
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    head_config: HeadConfig = field(default_factory=HeadConfig)
    streaming_config: StreamingConfig = field(default_factory=StreamingConfig)
    chunk_length: int = 340
    chunk_right_context: int = 40
    chunk_left_context: int = 0
    fifo_length: int = 40
    speaker_cache_update_period: int = 300

    @classmethod
    def from_dict(cls, value: dict):
        if value.get("model_type", "nemotron3_diarization") != "nemotron3_diarization":
            raise ValueError("Expected a Nemotron 3 Diarization checkpoint")
        args = {f.name: value[f.name] for f in fields(cls) if f.name in value}
        for key, typ in [
            ("audio_config", AudioConfig),
            ("head_config", HeadConfig),
            ("streaming_config", StreamingConfig),
        ]:
            args[key] = _read(typ, value.get(key))
        return cls(**args)

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return {"model_type": "nemotron3_diarization", **asdict(self)}

    def __post_init__(self):
        a, h, s = self.audio_config, self.head_config, self.streaming_config
        for obj in (a, h, s):
            for f in fields(obj):
                v = getattr(obj, f.name)
                if isinstance(v, int) and not isinstance(v, bool):
                    if v < 0 or (v == 0 and f.name != "speaker_cache_silence_frames_per_speaker"):
                        raise ValueError(f"{f.name} must be positive")
        if a.hidden_size % a.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if a.num_key_value_heads not in (None, a.num_attention_heads):
            raise ValueError(
                "This implementation supports the checkpoint's equal Q/K/V head counts"
            )
        if a.hidden_act != "gelu":
            raise ValueError("Only exact GELU is supported")
        if a.rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("Only default RoPE is supported")
        rotary_dim = int(
            a.hidden_size
            // a.num_attention_heads
            * a.rope_parameters.get("partial_rotary_factor", 1.0)
        )
        if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > a.hidden_size // a.num_attention_heads:
            raise ValueError("RoPE dimension must be positive, even and no larger than head_dim")
        if a.rope_parameters.get("rope_theta", 10000.0) <= 0:
            raise ValueError("rope_theta must be positive")
        if h.audio_hidden_size != a.hidden_size:
            raise ValueError("Encoder and head hidden dimensions disagree")
        if not (a.subsampling_factor == h.subsampling_factor == s.subsampling_factor):
            raise ValueError("Encoder, head and cache subsampling factors disagree")
        if h.num_speakers != s.num_speakers:
            raise ValueError("Head and cache speaker counts disagree")
        if (
            s.speaker_cache_length
            < (1 + s.speaker_cache_silence_frames_per_speaker) * s.num_speakers
        ):
            raise ValueError("Speaker cache is too small")
        if not 0 < s.prediction_score_threshold <= 0.5:
            raise ValueError("Invalid cache probability threshold")
        for name in ("strong_boost_rate", "weak_boost_rate", "min_positive_scores_rate"):
            if getattr(s, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.chunk_length < 1 or self.fifo_length < 1 or self.speaker_cache_update_period < 1:
            raise ValueError("Offline chunk, FIFO and update lengths must be positive")
        if self.chunk_left_context < 0 or self.chunk_right_context < 0:
            raise ValueError("Context lengths cannot be negative")
