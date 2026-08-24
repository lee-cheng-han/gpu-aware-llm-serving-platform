import asyncio
import time
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from inference.backend import BackendBundle, BackendExecutor
from inference.token_engine import EngineCapacityError, TokenGenerationEngine
from inference.token_types import (
    FinishReason,
    GeneratedToken,
    GenerationComplete,
    SamplingParameters,
    TokenizedPrompt,
)
from runtime.base import RuntimeCapacity
from runtime.huggingface import HuggingFaceTokenBackend
from serving_platform.domain import DeviceType


class FakeTokenBackend:
    def __init__(self, tokens=(10, 11, 12), eos_ids=frozenset({99})):
        self.tokens = list(tokens)
        self.eos_ids = eos_ids
        self.released: list[str] = []
        self.prefill_entered = Event()
        self.prefill_continue = Event()
        self.block_prefill = False
        self.decode_entered = Event()
        self.decode_continue = Event()
        self.block_decode = False
        self.fail_decode = False
        self.closed = False

    def load_model(self):
        return None

    def unload_model(self):
        return None

    def tokenize(self, prompt):
        return TokenizedPrompt((1, 2, 3))

    def prefill(self, sequence_id, prompt, parameters):
        self.prefill_entered.set()
        if self.block_prefill:
            self.prefill_continue.wait(1)
        return GeneratedToken(self.tokens[0], f"t{self.tokens[0]}", 0, len(prompt.token_ids))

    def decode(self, sequence_id, parameters, index):
        self.decode_entered.set()
        if self.block_decode:
            self.decode_continue.wait(1)
        if self.fail_decode:
            raise RuntimeError("controlled decode failure")
        token_id = self.tokens[index]
        return GeneratedToken(token_id, f"t{token_id}", index, 3 + index)

    def release_sequence(self, sequence_id):
        self.released.append(sequence_id)

    def capacity(self):
        return RuntimeCapacity(DeviceType.CPU, "cpu", None, None)

    def shutdown(self):
        self.closed = True

    def vocabulary_size(self):
        return 100

    def eos_token_ids(self):
        return self.eos_ids


def engine(backend, maximum=1):
    return TokenGenerationEngine(
        BackendBundle(backend, BackendExecutor("test-token-backend")), maximum
    )


def test_cuda_configuration_never_falls_back_to_cpu(monkeypatch):
    class UnavailableCudaWorker:
        device = "cuda:0"

        def warmup(self):
            pytest.fail("worker warmup must not run without CUDA")

    monkeypatch.setattr(
        "runtime.huggingface.token_backend.import_module",
        lambda name: SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        ),
    )
    backend = HuggingFaceTokenBackend(UnavailableCudaWorker())

    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        backend.load_model()


async def collect(engine, request_id):
    return [event async for event in engine.stream(request_id)]


async def test_sequence_prefills_decodes_completes_and_releases():
    backend = FakeTokenBackend()
    token_engine = engine(backend)
    token_engine.reserve("request", "hello", SamplingParameters(3), time.monotonic() + 10)

    events = await collect(token_engine, "request")

    assert [event.token_id for event in events if isinstance(event, GeneratedToken)] == [
        10,
        11,
        12,
    ]
    complete = events[-1]
    assert isinstance(complete, GenerationComplete)
    assert complete.finish_reason == FinishReason.LENGTH
    assert complete.input_tokens == 3
    assert complete.output_tokens == 3
    assert backend.released == ["request"]
    assert token_engine.active_sequence_count() == 0
    await token_engine.shutdown()


@pytest.mark.parametrize(
    ("tokens", "stops", "eos", "expected"),
    [
        ((99,), (), frozenset({99}), FinishReason.EOS),
        ((7,), (7,), frozenset({99}), FinishReason.STOP),
    ],
)
async def test_terminal_token_is_not_emitted(tokens, stops, eos, expected):
    backend = FakeTokenBackend(tokens=tokens, eos_ids=eos)
    token_engine = engine(backend)
    token_engine.reserve(
        "request",
        "hello",
        SamplingParameters(3, stop_token_ids=stops),
        time.monotonic() + 10,
    )

    events = await collect(token_engine, "request")

    assert len(events) == 1
    assert events[0].finish_reason == expected
    assert events[0].output_tokens == 0
    assert backend.released == ["request"]
    await token_engine.shutdown()


async def test_cancellation_during_prefill_emits_no_token_and_cleans_up():
    backend = FakeTokenBackend()
    backend.block_prefill = True
    token_engine = engine(backend)
    token_engine.reserve("request", "hello", SamplingParameters(2), time.monotonic() + 10)
    task = asyncio.create_task(collect(token_engine, "request"))
    await asyncio.to_thread(backend.prefill_entered.wait, 1)

    assert token_engine.cancel("request")
    backend.prefill_continue.set()
    events = await task

    assert len(events) == 1
    assert events[0].finish_reason == FinishReason.CANCELLED
    assert backend.released == ["request"]
    await token_engine.shutdown()


async def test_task_cancellation_during_prefill_waits_then_releases_cache():
    backend = FakeTokenBackend()
    backend.block_prefill = True
    token_engine = engine(backend)
    token_engine.reserve("request", "hello", SamplingParameters(2), time.monotonic() + 10)
    task = asyncio.create_task(collect(token_engine, "request"))
    await asyncio.to_thread(backend.prefill_entered.wait, 1)

    task.cancel()
    backend.prefill_continue.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert backend.released == ["request"]
    assert token_engine.active_sequence_count() == 0
    await token_engine.shutdown()


async def test_decode_failure_releases_sequence_without_terminal_emission():
    backend = FakeTokenBackend()
    backend.fail_decode = True
    token_engine = engine(backend)
    token_engine.reserve("request", "hello", SamplingParameters(2), time.monotonic() + 10)

    with pytest.raises(RuntimeError, match="controlled decode failure"):
        await collect(token_engine, "request")

    assert backend.released == ["request"]
    assert token_engine.active_sequence_count() == 0
    await token_engine.shutdown()


async def test_cancellation_during_decode_emits_nothing_after_active_operation():
    backend = FakeTokenBackend()
    backend.block_decode = True
    token_engine = engine(backend)
    token_engine.reserve("request", "hello", SamplingParameters(2), time.monotonic() + 10)
    task = asyncio.create_task(collect(token_engine, "request"))
    await asyncio.to_thread(backend.decode_entered.wait, 1)

    assert token_engine.cancel("request")
    backend.decode_continue.set()
    events = await task

    assert [event.token_id for event in events if isinstance(event, GeneratedToken)] == [10]
    assert events[-1].finish_reason == FinishReason.CANCELLED
    assert backend.released == ["request"]
    await token_engine.shutdown()


async def test_expired_deadline_times_out_before_prefill_and_releases_capacity():
    backend = FakeTokenBackend()
    token_engine = engine(backend)
    token_engine.reserve("request", "hello", SamplingParameters(2), time.monotonic() - 1)

    events = await collect(token_engine, "request")

    assert len(events) == 1
    assert events[0].finish_reason == FinishReason.TIMEOUT
    assert backend.released == []
    assert token_engine.active_sequence_count() == 0
    await token_engine.shutdown()


async def test_engine_capacity_is_reserved_before_streaming_starts():
    token_engine = engine(FakeTokenBackend(), maximum=1)
    token_engine.reserve("one", "hello", SamplingParameters(1), time.monotonic() + 10)

    with pytest.raises(EngineCapacityError):
        token_engine.reserve("two", "hello", SamplingParameters(1), time.monotonic() + 10)

    token_engine.cancel("one")
    await collect(token_engine, "one")
    await token_engine.shutdown()


async def test_backend_executor_serializes_commands():
    executor = BackendExecutor("serialization-test")
    lock = Lock()
    active = peak = 0

    def command():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1

    await asyncio.gather(
        executor.run_async(command),
        executor.run_async(command),
        executor.run_async(command),
    )
    executor.shutdown()
    assert peak == 1

    with pytest.raises(RuntimeError, match="shut down"):
        executor.run(command)
