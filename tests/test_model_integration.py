import asyncio
import json
import os

import httpx
import pytest

from apps.gateway.config import Settings
from apps.gateway.main import create_app
from inference.backend import BackendBundle, BackendExecutor
from inference.token_engine import TokenGenerationEngine
from inference.token_types import SamplingParameters
from inference.worker import InferenceWorker
from runtime.huggingface import HuggingFaceTokenBackend

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.getenv("RUN_MODEL_TESTS") != "1",
        reason="set RUN_MODEL_TESTS=1 to download and execute reference models",
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


def test_smollm_manual_prefill_decode_matches_greedy_generate():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    worker = InferenceWorker(model_id, revision, "cpu", "float32")
    backend = HuggingFaceTokenBackend(worker)
    executor = BackendExecutor("model-integration")
    prompt = "Explain cache memory in one sentence."
    parameters = SamplingParameters(3, temperature=0)
    try:
        tokenized = executor.run(backend.tokenize, prompt)
        import torch

        encoded = worker.tokenizer(prompt, return_tensors="pt")
        with worker._execution_lock, torch.inference_mode():
            expected = worker.model.generate(
                **encoded,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=worker.tokenizer.pad_token_id,
            )[0, encoded["input_ids"].shape[1]:].tolist()

        first = executor.run(backend.prefill, "manual", tokenized, parameters)
        tokens = [first]
        for index in range(1, 3):
            tokens.append(executor.run(backend.decode, "manual", parameters, index))

        assert [token.token_id for token in tokens] == expected
        assert tokens[0].cache_tokens == len(tokenized.token_ids)
        assert tokens[-1].cache_tokens > tokens[0].cache_tokens
    finally:
        executor.run(backend.release_sequence, "manual")
        executor.run(backend.shutdown)
        executor.shutdown()


async def test_smollm_sse_api_honors_eos_and_stop_tokens():
    class FixedSampler:
        def __init__(self, token_id):
            self.token_id = token_id

        def select(self, logits, parameters):
            return self.token_id

    worker = InferenceWorker(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "cpu",
        "float32",
    )
    sampler = FixedSampler(0)
    backend = HuggingFaceTokenBackend(worker, sampler)
    executor = BackendExecutor("model-stream-integration")
    token_engine = TokenGenerationEngine(BackendBundle(backend, executor))
    app = create_app(
        Settings(
            model_name=worker.model_name,
            model_revision=worker.revision,
            model_device="cpu",
            model_dtype="float32",
        ),
        worker,
        token_engine=token_engine,
    )

    def events(response):
        return [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]

    try:
        eos_id = next(iter(await executor.run_async(backend.eos_token_ids)))
        sampler.token_id = eos_id
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            eos_response = await asyncio.wait_for(
                client.post(
                    "/v1/generate_stream",
                    json={"prompt": "Hello", "max_new_tokens": 2, "temperature": 0},
                ),
                timeout=30,
            )
            eos_events = events(eos_response)
            assert eos_response.status_code == 200
            assert eos_events[-1]["finish_reason"] == "eos"
            assert eos_events[-1]["usage"]["output_tokens"] == 0

            stop_id = 10
            sampler.token_id = stop_id
            stop_response = await asyncio.wait_for(
                client.post(
                    "/v1/generate_stream",
                    json={
                        "prompt": "Hello",
                        "max_new_tokens": 2,
                        "temperature": 0,
                        "stop_token_ids": [stop_id],
                    },
                ),
                timeout=30,
            )
            stop_events = events(stop_response)
            assert stop_response.status_code == 200
            assert stop_events[-1]["finish_reason"] == "stop"
            assert stop_events[-1]["usage"]["output_tokens"] == 0
        assert backend.active_sequence_count() == 0
    finally:
        await asyncio.wait_for(token_engine.shutdown(), timeout=30)
