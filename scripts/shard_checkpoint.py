"""Split model.safetensors into bounded shards without torch or transformers.

Useful when transferring through a channel with a per-file size limit.
config.json and processor_config.json are copied alongside the shard index.
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Directory containing model.safetensors and JSON files")
    parser.add_argument("output")
    parser.add_argument("--max-mib", type=int, default=24)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    limit = args.max_mib * 1024**2
    if limit < 1024**2:
        parser.error("--max-mib must be positive")
    if source.resolve() == output.resolve():
        parser.error("Use a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.safetensors")) or (output / "model.safetensors.index.json").exists():
        parser.error("Output already contains checkpoint files")
    weight_map, total, part, pending, pending_size = {}, 0, 0, {}, 0

    def flush():
        nonlocal part, pending, pending_size
        if not pending:
            return
        part += 1
        name = f"model-{part:05d}.safetensors"
        save_file(pending, output / name, metadata={"format": "pt"})
        if (output / name).stat().st_size >= limit:
            raise ValueError("Shard exceeded requested size including its header")
        weight_map.update({key: name for key in pending})
        pending, pending_size = {}, 0

    with safe_open(source / "model.safetensors", framework="numpy") as checkpoint:
        for key in checkpoint.keys():
            value = checkpoint.get_tensor(key)
            if value.nbytes + 65536 >= limit:
                raise ValueError(f"Tensor {key} is too large for an individual shard")
            if pending_size + value.nbytes + 65536 >= limit:
                flush()
            pending[key] = value
            pending_size += value.nbytes
            total += value.nbytes
        flush()
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total},
                "weight_map": weight_map,
            },
            indent=2,
        )
        + "\n"
    )
    for name in ("config.json", "processor_config.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, output / name)
    print(f"Wrote {part} shards, each smaller than {args.max_mib} MiB, to {output}")


if __name__ == "__main__":
    main()
