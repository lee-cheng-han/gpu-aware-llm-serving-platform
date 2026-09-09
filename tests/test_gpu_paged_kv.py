import json
import os
import platform
from pathlib import Path

import pytest

from inference.paged_kv import PagedKVConfig, TorchPagedKVCache

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.getenv("RUN_GPU_TESTS") != "1",
        reason="set RUN_GPU_TESTS=1 on a CUDA host",
    ),
]


def test_physical_paged_kv_uses_cuda_and_batches_different_lengths():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.fail("RUN_GPU_TESTS=1 requires CUDA; CPU fallback is forbidden")
    before = torch.cuda.memory_allocated(0)
    cache = TorchPagedKVCache(
        PagedKVConfig(16, 4, 2, 2, 8, device="cuda:0", dtype="float16")
    )
    after_allocation = torch.cuda.memory_allocated(0)
    cache.allocate("short", 3)
    cache.allocate("long", 7)
    short = tuple(torch.randn(2, 3, 8, device="cuda:0", dtype=torch.float16) for _ in range(2))
    long = tuple(torch.randn(2, 7, 8, device="cuda:0", dtype=torch.float16) for _ in range(2))
    cache.append("short", short, short)
    cache.append("long", long, long)
    keys, values, mask = cache.materialize(0, ("short", "long"))

    assert cache.key_pages.device.type == cache.value_pages.device.type == "cuda"
    assert keys.device.type == values.device.type == mask.device.type == "cuda"
    assert keys.shape == (2, 2, 7, 8)
    assert mask[:, 0, 0, :].sum(dim=1).tolist() == [3, 7]
    assert after_allocation > before
    cache.assert_invariants()

    output = os.getenv("PAGED_KV_VALIDATION_OUTPUT")
    if output:
        report = {
            "verified": True,
            "classification": "measured_cuda_correctness_not_performance",
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "page_layout": list(cache.key_pages.shape),
            "different_lengths": [3, 7],
            "allocated_memory_before_bytes": before,
            "allocated_memory_after_bytes": after_allocation,
        }
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
