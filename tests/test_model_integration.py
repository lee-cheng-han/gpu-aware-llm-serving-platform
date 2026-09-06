import asyncio
import json
import os
import time

import httpx
import pytest

from apps.gateway.config import Settings
from apps.gateway.main import create_app
from inference.backend import BackendBundle, BackendExecutor
from inference.token_engine import TokenGenerationEngine
from inference.token_types import SamplingParameters, TokenizedPrompt
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


def test_smollm_prefix_snapshot_restores_identical_greedy_state():
    worker = InferenceWorker(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "cpu",
        "float32",
    )
    backend = HuggingFaceTokenBackend(worker)
    executor = BackendExecutor("model-prefix-integration")
    parameters = SamplingParameters(2, temperature=0)
    try:
        prompt = executor.run(backend.tokenize, "Explain prefix caching briefly.")
        original = executor.run(backend.prefill, "original", prompt, parameters)
        snapshot = executor.run(backend.snapshot_prefix, "original")
        restored = executor.run(
            backend.restore_prefix, "restored", snapshot, parameters
        )
        assert restored.token_id == original.token_id
        assert restored.cache_tokens == original.cache_tokens == len(prompt.token_ids)
        original_next = executor.run(backend.decode, "original", parameters, 1)
        restored_next = executor.run(backend.decode, "restored", parameters, 1)
        assert restored_next.token_id == original_next.token_id
        assert restored_next.cache_tokens == original_next.cache_tokens
    finally:
        executor.run(backend.release_sequence, "original")
        executor.run(backend.release_sequence, "restored")
        executor.run(backend.shutdown)
        executor.shutdown()


def test_smollm_partial_prefix_resume_matches_full_prefill():
    worker = InferenceWorker(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "cpu",
        "float32",
    )
    backend = HuggingFaceTokenBackend(worker)
    executor = BackendExecutor("model-partial-prefix-integration")
    parameters = SamplingParameters(2, temperature=0)
    try:
        full = executor.run(
            backend.tokenize,
            "Explain why cached prefixes reduce repeated prompt work.",
        )
        split = max(1, len(full.token_ids) // 2)
        prefix = TokenizedPrompt(full.token_ids[:split])
        suffix = TokenizedPrompt(full.token_ids[split:])
        executor.run(backend.prefill, "prefix", prefix, parameters)
        snapshot = executor.run(backend.snapshot_prefix, "prefix")
        resumed = executor.run(
            backend.resume_prefix,
            "resumed",
            snapshot,
            suffix,
            parameters,
        )
        reference = executor.run(backend.prefill, "reference", full, parameters)
        assert resumed.token_id == reference.token_id
        assert resumed.cache_tokens == reference.cache_tokens == len(full.token_ids)
        resumed_next = executor.run(backend.decode, "resumed", parameters, 1)
        reference_next = executor.run(backend.decode, "reference", parameters, 1)
        assert resumed_next.token_id == reference_next.token_id
    finally:
        for sequence_id in ("prefix", "resumed", "reference"):
            executor.run(backend.release_sequence, sequence_id)
        executor.run(backend.shutdown)
        executor.shutdown()


def test_smollm_real_batched_decode_and_chunked_prefill_match_reference():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    worker = InferenceWorker(model_id, revision, "cpu", "float32")
    backend = HuggingFaceTokenBackend(worker)
    executor = BackendExecutor("model-batch-integration")
    prompts = ("Hello", "Explain cache memory")
    parameters = SamplingParameters(2, temperature=0)
    try:
        tokenized = [executor.run(backend.tokenize, prompt) for prompt in prompts]
        import torch

        encoded = worker.tokenizer(list(prompts), return_tensors="pt", padding=True)
        with worker._execution_lock, torch.inference_mode():
            expected = worker.model.generate(
                **encoded,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=worker.tokenizer.pad_token_id,
            )[:, encoded["input_ids"].shape[1] :].tolist()

        prefill = executor.run(
            backend.prefill_batch,
            tuple(
                (f"batch-{index}", prompt, parameters)
                for index, prompt in enumerate(tokenized)
            ),
        )
        decoded = executor.run(
            backend.decode_batch,
            tuple((f"batch-{index}", parameters, 1) for index in range(2)),
        )
        actual = [
            [
                prefill.tokens[f"batch-{index}"].token_id,
                decoded.tokens[f"batch-{index}"].token_id,
            ]
            for index in range(2)
        ]
        assert actual == expected

        full_prompt = tokenized[1]
        first_chunk = TokenizedPrompt(full_prompt.token_ids[:1])
        second_chunk = TokenizedPrompt(full_prompt.token_ids[1:])
        assert executor.run(
            backend.prefill_chunk, "chunked", first_chunk, parameters, False
        ) is None
        chunked = executor.run(
            backend.prefill_chunk, "chunked", second_chunk, parameters, True
        )
        assert chunked is not None
        assert chunked.token_id == expected[1][0]
        assert chunked.cache_tokens == len(full_prompt.token_ids)
    finally:
        for sequence_id in ("batch-0", "batch-1", "chunked"):
            executor.run(backend.release_sequence, sequence_id)
        executor.run(backend.shutdown)
        executor.shutdown()


async def test_smollm_continuous_engine_batches_concurrent_sequences():
    worker = InferenceWorker(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "cpu",
        "float32",
    )
    backend = HuggingFaceTokenBackend(worker)
    engine = TokenGenerationEngine(
        BackendBundle(backend, BackendExecutor("real-continuous-engine")),
        max_active_sequences=2,
        max_batch_size=2,
        scheduling_policy="balanced",
    )
    prompts = {"first": "Hello", "second": "World"}
    try:
        for request_id, prompt in prompts.items():
            prompt_tokens = await engine.count_prompt_tokens(prompt)
            engine.reserve(
                request_id,
                prompt,
                SamplingParameters(2, temperature=0),
                time.monotonic() + 30,
                prompt_tokens,
            )

        async def collect(request_id):
            return [event async for event in engine.stream(request_id)]

        first, second = await asyncio.gather(collect("first"), collect("second"))
        assert len(first) == len(second) == 3
        assert first[-1].finish_reason.value == "length"
        assert second[-1].finish_reason.value == "length"
        snapshot = engine.scheduler_snapshot()
        assert snapshot.prefill_batches == 1
        assert snapshot.decode_batches == 1
        assert snapshot.max_observed_batch_size == 2
        assert snapshot.kv_cache.used_blocks == 0
    finally:
        await engine.shutdown()


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
