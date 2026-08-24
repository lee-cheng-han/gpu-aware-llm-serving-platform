import json
import os
import platform
import subprocess
from pathlib import Path

import pytest

from inference.backend import BackendExecutor
from inference.token_types import SamplingParameters
from inference.worker import InferenceWorker
from runtime.huggingface import HuggingFaceTokenBackend

pytestmark = [
    pytest.mark.model,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.getenv("RUN_GPU_TESTS") != "1",
        reason="set RUN_GPU_TESTS=1 on a CUDA host",
    ),
]


def test_real_cuda_prefill_decode_and_emit_validation_report():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.fail("RUN_GPU_TESTS=1 requires CUDA; CPU fallback is forbidden")
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    before = torch.cuda.memory_allocated(0)
    worker = InferenceWorker(model_id, revision, "cuda:0", "float16")
    backend = HuggingFaceTokenBackend(worker)
    executor = BackendExecutor("cuda-validation")
    try:
        tokenized = executor.run(backend.tokenize, "Explain GPU scheduling briefly.")
        token = executor.run(
            backend.prefill,
            "cuda-proof",
            tokenized,
            SamplingParameters(2, temperature=0),
        )
        second = executor.run(
            backend.decode,
            "cuda-proof",
            SamplingParameters(2, temperature=0),
            1,
        )
        capacity = executor.run(backend.capacity)
        assert worker.model.device.type == "cuda"
        sequence_cache = backend._sequences["cuda-proof"]
        assert sequence_cache.attention_mask.device.type == "cuda"
        assert sequence_cache.cache_position.device.type == "cuda"
        assert all(
            tensor.device.type == "cuda"
            for tensor in (
                *sequence_cache.cache.key_cache,
                *sequence_cache.cache.value_cache,
            )
        )
        assert torch.cuda.memory_allocated(0) > before
        assert second.cache_tokens > token.cache_tokens
        assert capacity.device_type.value == "cuda"

        cpu_worker = InferenceWorker(model_id, revision, "cpu", "float32")
        cpu_worker.warmup()
        encoded = cpu_worker.tokenizer(
            "Explain GPU scheduling briefly.", return_tensors="pt"
        )
        with torch.inference_mode():
            expected = cpu_worker.model.generate(
                **encoded,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=cpu_worker.tokenizer.pad_token_id,
            )[0, encoded["input_ids"].shape[1] :].tolist()
        assert [token.token_id, second.token_id] == expected
        try:
            driver = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).splitlines()[0]
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
            driver = "unavailable"
        report = {
            "verified": True,
            "model_id": model_id,
            "model_revision": revision,
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "driver": driver,
            "device": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "input_tokens": len(tokenized.token_ids),
            "output_token_ids": [token.token_id, second.token_id],
            "cpu_greedy_token_ids": expected,
            "cache_tensors_on_cuda": True,
            "allocated_memory_bytes": torch.cuda.memory_allocated(0),
        }
        output = os.getenv("GPU_VALIDATION_OUTPUT")
        if output:
            Path(output).write_text(json.dumps(report, indent=2) + "\n")
    finally:
        executor.run(backend.release_sequence, "cuda-proof")
        executor.run(backend.shutdown)
        executor.shutdown()
