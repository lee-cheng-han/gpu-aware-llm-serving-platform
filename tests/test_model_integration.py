import os

import pytest

from inference.worker import InferenceWorker

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.getenv("RUN_MODEL_TESTS") != "1",
        reason="set RUN_MODEL_TESTS=1 to download and execute Tiny GPT-2",
    ),
]


def test_tiny_gpt2_generation_batch_and_stream():
    worker = InferenceWorker("sshleifer/tiny-gpt2")
    worker.warmup()
    assert worker.is_ready
    assert worker.context_window_tokens() > 0

    single = worker.generate_one("Local inference", 2, 0)
    assert single.input_tokens > 0
    assert 0 < single.output_tokens <= 2

    batch = worker.generate_batch(["First prompt", "Second prompt"], 2, 0)
    assert len(batch) == 2
    assert all(0 < result.output_tokens <= 2 for result in batch)

    chunks = list(worker.stream("Stream locally", 2, 0))
    assert chunks
    assert 0 < chunks[-1].output_tokens <= 2
