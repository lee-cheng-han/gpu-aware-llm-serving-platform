import asyncio
import time
from threading import Event

import pytest

from inference.backend import BackendBundle, BackendExecutor, BatchTokenResult
from inference.kv_cache import KVBlockAllocator
from inference.prefix_cache import PrefixCache
from inference.token_engine import EngineCapacityError, TokenGenerationEngine
from inference.token_types import (
    FinishReason,
    GeneratedToken,
    GenerationComplete,
    SamplingParameters,
    TokenizedPrompt,
)
from runtime.base import RuntimeCapacity
from serving_platform.domain import DeviceType


class BatchBackend:
    def __init__(self):
        self.positions = {}
        self.prefill_batches = []
        self.decode_batches = []
        self.released = []
        self.fail_on_decode = set()
        self.decode_entered = Event()
        self.decode_continue = Event()
        self.block_first_decode = False
        self.prefix_snapshots = 0
        self.prefix_restores = 0

    def load_model(self):
        return None

    def unload_model(self):
        return None

    def tokenize(self, prompt):
        return TokenizedPrompt(tuple(range(1, int(prompt) + 1)))

    def prefill(self, sequence_id, prompt, parameters):
        return self.prefill_batch(((sequence_id, prompt, parameters),)).tokens[sequence_id]

    def prefill_batch(self, requests):
        self.prefill_batches.append(tuple(item[0] for item in requests))
        tokens = {}
        for sequence_id, prompt, _ in requests:
            self.positions[sequence_id] = len(prompt.token_ids)
            tokens[sequence_id] = GeneratedToken(10, "a", 0, len(prompt.token_ids))
        return BatchTokenResult(tokens)

    def decode(self, sequence_id, parameters, index):
        return self.decode_batch(((sequence_id, parameters, index),)).tokens[sequence_id]

    def decode_batch(self, requests):
        self.decode_batches.append(tuple(item[0] for item in requests))
        if self.block_first_decode and len(self.decode_batches) == 1:
            self.decode_entered.set()
            self.decode_continue.wait(2)
        tokens = {}
        failures = {}
        for sequence_id, _, index in requests:
            if sequence_id in self.fail_on_decode:
                failures[sequence_id] = RuntimeError(f"failed {sequence_id}")
                continue
            self.positions[sequence_id] += 1
            tokens[sequence_id] = GeneratedToken(
                10 + index, chr(97 + index), index, self.positions[sequence_id]
            )
        return BatchTokenResult(tokens, failures)

    def release_sequence(self, sequence_id):
        self.released.append(sequence_id)

    def snapshot_prefix(self, sequence_id):
        self.prefix_snapshots += 1
        return self.positions[sequence_id]

    def restore_prefix(self, sequence_id, snapshot, parameters):
        self.prefix_restores += 1
        self.positions[sequence_id] = snapshot
        return GeneratedToken(10, "a", 0, snapshot)

    def capacity(self):
        return RuntimeCapacity(DeviceType.CPU, "cpu", None, None)

    def shutdown(self):
        return None

    def vocabulary_size(self):
        return 100

    def eos_token_ids(self):
        return frozenset({99})


def make_engine(backend, **kwargs):
    return TokenGenerationEngine(
        BackendBundle(backend, BackendExecutor("continuous-test")),
        max_active_sequences=4,
        max_batch_size=4,
        scheduling_policy="balanced",
        kv_allocator=KVBlockAllocator(32, 2),
        **kwargs,
    )


async def collect(engine, request_id):
    return [event async for event in engine.stream(request_id)]


async def wait_until(event):
    for _ in range(1000):
        if event.is_set():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("timed out waiting for backend event")


async def test_concurrent_sequences_share_prefill_and_decode_iterations():
    backend = BatchBackend()
    engine = make_engine(backend)
    for request_id in ("first", "second"):
        engine.reserve(
            request_id, "3", SamplingParameters(3), time.monotonic() + 10, 3
        )

    first, second = await asyncio.gather(
        collect(engine, "first"), collect(engine, "second")
    )

    assert backend.prefill_batches == [("first", "second")]
    assert backend.decode_batches == [
        ("first", "second"),
        ("first", "second"),
    ]
    assert [event.token_id for event in first[:-1]] == [10, 11, 12]
    assert [event.token_id for event in second[:-1]] == [10, 11, 12]
    assert all(event[-1].finish_reason == FinishReason.LENGTH for event in (first, second))
    snapshot = engine.scheduler_snapshot()
    assert snapshot.max_observed_batch_size == 2
    assert snapshot.kv_cache.used_blocks == 0
    await engine.shutdown()


async def test_sequence_failure_does_not_corrupt_batch_peer():
    backend = BatchBackend()
    backend.fail_on_decode.add("broken")
    engine = make_engine(backend)
    for request_id in ("broken", "healthy"):
        engine.reserve(
            request_id, "2", SamplingParameters(2), time.monotonic() + 10, 2
        )

    broken_task = asyncio.create_task(collect(engine, "broken"))
    healthy_task = asyncio.create_task(collect(engine, "healthy"))
    with pytest.raises(RuntimeError, match="failed broken"):
        await broken_task
    healthy = await healthy_task

    assert isinstance(healthy[-1], GenerationComplete)
    assert healthy[-1].finish_reason == FinishReason.LENGTH
    assert set(backend.released) == {"broken", "healthy"}
    assert engine.scheduler_snapshot().kv_cache.used_blocks == 0
    await engine.shutdown()


async def test_new_request_joins_while_an_existing_sequence_is_decoding():
    backend = BatchBackend()
    backend.block_first_decode = True
    engine = make_engine(backend)
    engine.reserve("existing", "2", SamplingParameters(6), time.monotonic() + 10, 2)
    existing_task = asyncio.create_task(collect(engine, "existing"))
    await wait_until(backend.decode_entered)

    engine.reserve("joining", "2", SamplingParameters(4), time.monotonic() + 10, 2)
    joining_task = asyncio.create_task(collect(engine, "joining"))
    backend.decode_continue.set()
    existing, joining = await asyncio.gather(existing_task, joining_task)

    assert backend.prefill_batches == [("existing",), ("joining",)]
    assert ("existing", "joining") in backend.decode_batches
    assert existing[-1].finish_reason == FinishReason.LENGTH
    assert joining[-1].finish_reason == FinishReason.LENGTH
    assert engine.scheduler_snapshot().prefill_batches == 2
    await engine.shutdown()


async def test_kv_exhaustion_rejects_before_streaming_and_rolls_back():
    backend = BatchBackend()
    engine = TokenGenerationEngine(
        BackendBundle(backend, BackendExecutor("kv-capacity-test")),
        4,
        kv_allocator=KVBlockAllocator(2, 2),
    )
    engine.reserve("fits", "2", SamplingParameters(2), time.monotonic() + 10, 2)
    with pytest.raises(EngineCapacityError, match="KV cache"):
        engine.reserve("full", "2", SamplingParameters(2), time.monotonic() + 10, 2)
    assert engine.scheduler_snapshot().kv_cache.used_blocks == 2
    engine.cancel("fits")
    await collect(engine, "fits")
    assert engine.scheduler_snapshot().kv_cache.used_blocks == 0
    await engine.shutdown()


async def test_cancelled_sequence_leaves_without_stopping_peer():
    backend = BatchBackend()
    engine = make_engine(backend)
    for request_id in ("cancelled", "healthy"):
        engine.reserve(
            request_id, "2", SamplingParameters(2), time.monotonic() + 10, 2
        )
    engine.cancel("cancelled")
    cancelled, healthy = await asyncio.gather(
        collect(engine, "cancelled"), collect(engine, "healthy")
    )
    assert cancelled[-1].finish_reason == FinishReason.CANCELLED
    assert healthy[-1].finish_reason == FinishReason.LENGTH
    await engine.shutdown()


async def test_chunked_prefill_advances_prompt_in_bounded_steps():
    class ChunkBackend(BatchBackend):
        def __init__(self):
            super().__init__()
            self.chunks = []

        def prefill_chunk(self, sequence_id, prompt, parameters, final):
            self.chunks.append((sequence_id, len(prompt.token_ids), final))
            self.positions[sequence_id] = self.positions.get(sequence_id, 0) + len(
                prompt.token_ids
            )
            if not final:
                return None
            return GeneratedToken(10, "a", 0, self.positions[sequence_id])

    backend = ChunkBackend()
    engine = make_engine(backend, prefill_chunk_size_tokens=2)
    engine.reserve("chunked", "5", SamplingParameters(1), time.monotonic() + 10, 5)

    result = await collect(engine, "chunked")

    assert backend.chunks == [
        ("chunked", 2, False),
        ("chunked", 2, False),
        ("chunked", 1, True),
    ]
    assert [event.token_id for event in result[:-1]] == [10]
    assert result[-1].finish_reason == FinishReason.LENGTH
    await engine.shutdown()


async def test_exact_prefix_hit_bypasses_a_second_prefill():
    backend = BatchBackend()
    engine = make_engine(
        backend,
        prefix_cache=PrefixCache(max_entries=2, max_tokens=16),
    )
    engine.reserve("first", "3", SamplingParameters(1), time.monotonic() + 10, 3)
    first = await collect(engine, "first")
    engine.reserve("second", "3", SamplingParameters(1), time.monotonic() + 10, 3)
    second = await collect(engine, "second")

    assert backend.prefill_batches == [("first",)]
    assert backend.prefix_snapshots == 1
    assert backend.prefix_restores == 1
    assert first[-1].finish_reason == second[-1].finish_reason == FinishReason.LENGTH
    snapshot = engine.scheduler_snapshot().prefix_cache
    assert snapshot.hits == 1
    assert snapshot.misses == 1
    await engine.shutdown()
