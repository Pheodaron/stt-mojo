"""CPU float32 MAX graphs, with optional MEF export for the native Go runtime."""

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Weight, ops

from .config import ModelConfig


class MaxGraphs:
    def __init__(self, config: ModelConfig, weights: dict[str, np.ndarray], *, export_dir=None):
        self.config, self.weights = config, weights
        self.export_dir = export_dir
        self.device = CPU()
        self.session = InferenceSession(devices=[self.device])
        self._registry = {}
        self.embedding = self._compile_embedding()
        self.encoder = self._compile_encoder()

    def _weight(self, name):
        value = self.weights[name]
        self._registry[name] = value
        return Weight(name, DType.float32, value.shape, device=DeviceRef.CPU())

    def _linear(self, x, name, bias=True):
        y = ops.matmul(x, ops.transpose(self._weight(name + ".weight"), 0, 1))
        return y + self._weight(name + ".bias") if bias else y

    def _norm(self, x, name):
        # nn.LayerNorm defaults to eps=1e-5 in the upstream model.
        return ops.layer_norm(
            x, self._weight(name + ".weight"), self._weight(name + ".bias"), epsilon=1e-5
        )

    def _compile(self, graph):
        compiled = self.session.compile(graph)
        if self.export_dir is not None:
            compiled.export_mef(str(self.export_dir / (graph.name + ".mef")))
        return self.session.init(compiled, weights_registry=self._registry)

    def _compile_embedding(self):
        a = self.config.audio_config
        self._registry = {}
        with Graph(
            "nemotron_feature_stacking",
            input_types=[
                TensorType(
                    DType.float32,
                    ["batch", "blocks", a.subsampling_factor * a.num_mel_bins],
                    device=DeviceRef.CPU(),
                ),
            ],
        ) as graph:
            graph.output(
                self._linear(
                    graph.inputs[0].tensor, "model.audio_tower.embedder.projection", bias=False
                )
            )
        return self._compile(graph)

    @staticmethod
    def _rotate(x, cos, sin, rotary_dim):
        rotating = x[:, :, :, :rotary_dim]
        half = rotary_dim // 2
        swapped = ops.concat([-rotating[:, :, :, half:], rotating[:, :, :, :half]], axis=-1)
        result = rotating * cos + swapped * sin
        if rotary_dim < int(x.shape[-1]):
            result = ops.concat([result, x[:, :, :, rotary_dim:]], axis=-1)
        return result

    def _compile_encoder(self):
        a, h = self.config.audio_config, self.config.head_config
        d, n = a.hidden_size, a.num_attention_heads
        head_dim = d // n
        rotary_dim = int(head_dim * a.rope_parameters.get("partial_rotary_factor", 1.0))
        cpu = DeviceRef.CPU()
        self._registry = {}
        with Graph(
            "nemotron_encoder_and_head",
            input_types=[
                TensorType(DType.float32, ["batch", "frames", d], device=cpu),
                TensorType(DType.float32, [1, 1, "frames", rotary_dim], device=cpu),
                TensorType(DType.float32, [1, 1, "frames", rotary_dim], device=cpu),
                TensorType(DType.float32, ["batch", 1, 1, "frames"], device=cpu),
            ],
        ) as graph:
            x, cos, sin, mask = (v.tensor for v in graph.inputs)
            batch, time = x.shape[0], x.shape[1]
            x = self._norm(x, "model.audio_tower.input_layer_norm")
            for i in range(a.num_hidden_layers):
                base = f"model.audio_tower.layers.{i}"
                normed = self._norm(x, base + ".layer_norm1")
                q, k, v = [
                    ops.transpose(
                        ops.reshape(
                            self._linear(normed, base + ".self_attn." + p, bias=False),
                            [batch, time, n, head_dim],
                        ),
                        1,
                        2,
                    )
                    for p in ("q_proj", "k_proj", "v_proj")
                ]
                q = self._rotate(q, cos, sin, rotary_dim)
                k = self._rotate(k, cos, sin, rotary_dim)
                scores = ops.matmul(q, ops.transpose(k, 2, 3)) * (head_dim**-0.5) + mask
                attended = ops.matmul(ops.softmax(scores), v)
                attended = ops.reshape(ops.transpose(attended, 1, 2), [batch, time, d])
                x = x + self._linear(attended, base + ".self_attn.o_proj")
                normed = self._norm(x, base + ".layer_norm2")
                x = x + self._linear(
                    ops.gelu(self._linear(normed, base + ".mlp.fc1"), approximate="none"),
                    base + ".mlp.fc2",
                )
            x = self._norm(x, "model.audio_tower.layer_norm")
            x = self._linear(x, "model.proj")

            # Conv1d cross-correlation as one matmul over left/current/right taps.
            # The reshape after it is the original subpixel shuffle, not repeat_interleave.
            zero = x[:, :1, :] * 0.0
            left = ops.concat([zero, x[:, :-1, :]], axis=1)
            right = ops.concat([x[:, 1:, :], zero], axis=1)
            taps = ops.concat([left, x, right], axis=-1)
            kernel = np.ascontiguousarray(
                self.weights["model.upsampler.conv.weight"]
                .transpose(2, 1, 0)
                .reshape(3 * h.hidden_size, -1)
            )
            self.weights["__upsample_kernel"] = kernel
            x = ops.matmul(taps, self._weight("__upsample_kernel")) + self._weight(
                "model.upsampler.conv.bias"
            )
            x = ops.reshape(x, [batch, time * a.subsampling_factor, h.hidden_size])
            x = self._linear(ops.relu(x), "classifier.dense")
            graph.output(self._linear(ops.relu(x), "classifier.out_proj"))
        result = self._compile(graph)
        # Keep the derived matrix alive through the model's lifetime.
        self._upsample_kernel = self.weights.pop("__upsample_kernel")
        return result

    @staticmethod
    def _run(model, *values):
        buffers = [Buffer.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for v in values]
        # Copy: do not expose a runtime-owned output buffer that later executions can reuse.
        return np.from_dlpack(model.execute(*buffers)[0]).copy()

    def embed(self, features: np.ndarray):
        b, t, m = features.shape
        f = self.config.audio_config.subsampling_factor
        padding = -t % f
        padded = np.pad(features, ((0, 0), (0, padding), (0, 0))) if padding else features
        return self._run(self.embedding, padded.reshape(b, -1, m * f))

    def infer(self, embeddings: np.ndarray, valid_mask: np.ndarray | None = None):
        b, t, _ = embeddings.shape
        a = self.config.audio_config
        head_dim = a.hidden_size // a.num_attention_heads
        dim = int(head_dim * a.rope_parameters.get("partial_rotary_factor", 1.0))
        theta = a.rope_parameters.get("rope_theta", 10000.0)
        inv = 1.0 / (np.float32(theta) ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        angles = np.arange(t, dtype=np.float32)[:, None] * inv
        angles = np.concatenate([angles, angles], axis=-1)[None, None]
        if valid_mask is None:
            valid_mask = np.ones((b, t), dtype=bool)
        if valid_mask.shape != (b, t) or not np.all(np.any(valid_mask, axis=1)):
            raise ValueError("Attention mask must have at least one valid key per sample")
        bias = np.where(valid_mask[:, None, None, :], np.float32(0), np.finfo(np.float32).min)
        return self._run(self.encoder, embeddings, np.cos(angles), np.sin(angles), bias)
