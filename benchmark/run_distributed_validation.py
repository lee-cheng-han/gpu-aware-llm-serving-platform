from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from runtime.distributed import DistributedConfig, DistributedContext


def _driver_version() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate real multi-process NCCL collectives under torchrun"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    import torch

    config = DistributedConfig.from_environment()
    if config.world_size < 2:
        parser.error("run with torchrun --nproc-per-node=2 or greater")
    context = DistributedContext(config)
    try:
        status = context.initialize()
        tensor = torch.tensor(
            [config.rank + 1.0], device=f"cuda:{config.local_rank}"
        )
        context.all_reduce_sum(tensor)
        expected = config.world_size * (config.world_size + 1) / 2
        reduce_correct = tensor.item() == expected
        broadcast = torch.tensor(
            [42.0 if config.rank == 0 else -1.0],
            device=f"cuda:{config.local_rank}",
        )
        context.broadcast(broadcast)
        broadcast_correct = broadcast.item() == 42
        devices = context.all_gather_objects(torch.cuda.get_device_name(config.local_rank))
        context.barrier()
        passed = reduce_correct and broadcast_correct and len(devices) == config.world_size
        report = {
            "verified": passed,
            "classification": "measured_nccl_collective_correctness_not_model_performance",
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "driver": _driver_version(),
            "backend": status.backend,
            "world_size": status.world_size,
            "devices": devices,
            "all_reduce_expected": expected,
            "all_reduce_actual": tensor.item(),
            "broadcast_actual": broadcast.item(),
        }
        if config.rank == 0:
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered)
            print(rendered, end="")
        if args.check and not passed:
            raise SystemExit("distributed validation failed")
    finally:
        context.shutdown()


if __name__ == "__main__":
    main()
