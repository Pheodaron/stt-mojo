"""Test-only loader for the original NeMo cache methods without installing NeMo.

The caller supplies sortformer_modules.py from the revision recorded in NOTICE.
AST extraction preserves the upstream method bodies and avoids its unrelated
training, CUDA and framework imports. Nothing here is used by the Go service.
"""

import ast
import math
from types import SimpleNamespace

import numpy as np
import torch


def load_cache(source, *, speakers, hidden, capacity, fifo, update):
    methods = {
        "streaming_update",
        "_compress_spkcache",
        "_get_log_pred_scores",
        "_disable_low_scores",
        "_boost_topk_scores",
        "_get_topk_indices",
        "_gather_spkcache_and_preds",
    }
    tree = ast.parse(source.read_text())
    original = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SortformerModules"
    )
    reference = ast.ClassDef(
        name="Reference",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods],
    )
    module = ast.fix_missing_locations(ast.Module(body=[reference], type_ignores=[]))
    namespace = {"torch": torch, "math": math}
    exec(compile(module, str(source), "exec"), namespace)
    cache = namespace["Reference"]()
    cache.n_spk = speakers
    cache.spkcache_len = capacity
    cache.fifo_len = fifo
    cache.spkcache_update_period = update
    cache.spkcache_sil_frames_per_spk = 1
    cache.pred_score_threshold = 0.25
    cache.strong_boost_rate = 0.75
    cache.weak_boost_rate = 1.5
    cache.min_pos_scores_rate = 0.5
    cache.scores_boost_latest = 0.05
    cache.max_index = 1_000_000
    cache.use_learnable_sil_emb = True
    cache.learnable_sil_emb = torch.full((hidden,), -1.0)
    cache.training = False
    cache.log = False
    return cache


def update_fixtures(source, rng):
    cache = load_cache(source, speakers=2, hidden=4, capacity=8, fifo=3, update=2)
    state = SimpleNamespace(
        spkcache=torch.empty(1, 0, 4),
        fifo=torch.empty(1, 0, 4),
        spkcache_preds=torch.empty(1, 0, 2),
        spk_perm=None,
        spkcache_compressed=False,
        mean_sil_emb=torch.zeros(1, 4),
    )
    result = []
    for step in range(15):
        left, central, right = int(step > 0), 3, 1
        chunk = torch.from_numpy(rng.normal(size=(1, left + central + right, 4)).astype(np.float32))
        inputs = torch.cat([state.spkcache, state.fifo, chunk], dim=1)
        logits = torch.from_numpy(rng.normal(0, 2, (1, inputs.shape[1] * 2, 2)).astype(np.float32))
        probs = logits.sigmoid().reshape(1, -1, 2, 2).mean(dim=2)
        state, _ = cache.streaming_update(state, chunk, probs, lc=left, rc=right)
        result.append(
            dict(
                inputs=inputs.flatten().tolist(),
                logits=logits.flatten().tolist(),
                lookback=left,
                frames=central,
                cache=state.spkcache.flatten().tolist(),
                probs=state.spkcache_preds.flatten().tolist(),
                fifo=state.fifo.flatten().tolist(),
            )
        )
    return result
