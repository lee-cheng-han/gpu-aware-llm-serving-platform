from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from apps.worker import ManagedWorker
from runtime.base import RuntimeCapacity, RuntimeResult
from runtime.simulated_gpu import SimulatedGpuConfig, SimulatedGpuRuntime
from serving_platform.domain import (
    DeviceType,
    ModelDefinition,
    RequestRecord,
    RequestState,
    RuntimeType,
)
from serving_platform.registry import InMemoryWorkerRegistry


def model(model_id: str, memory: int = 150, idle: float = 10) -> ModelDefinition:
    return ModelDefinition(
        model_id, "main", RuntimeType.SIMULATED, memory, ("float16",), "float16",
        100, 200, True, True, 5, idle,
    )


class BlockingRuntime:
    runtime_type = RuntimeType.SIMULATED
    device_type = DeviceType.SIMULATED_GPU

    def __init__(self, definition: ModelDefinition, fail: bool = False):
        self.definition = definition
        self.fail = fail
        self.started = Event()
        self.release = Event()
        self.loaded = False
        self.load_calls = 0
        self._lock = Lock()

    def load_model(self, definition):
        with self._lock:
            self.load_calls += 1
        self.started.set()
        assert self.release.wait(1)
        if self.fail:
            raise RuntimeError("load failed")
        self.loaded = True

    def unload_model(self, model_id):
        self.loaded = False

    def warmup_model(self, model_id):
        if not self.loaded:
            raise RuntimeError("not loaded")

    def is_model_loaded(self, model_id):
        return self.loaded

    def count_prompt_tokens(self, model_id, prompt):
        return len(prompt.split())

    def generate(self, model_id, prompts, max_new_tokens, temperature):
        return [RuntimeResult("ok", 1, 1, 0, 0, 0) for _ in prompts]

    def stream(self, model_id, prompt, max_new_tokens, temperature):
        return iter(())

    def capacity(self):
        return RuntimeCapacity(DeviceType.SIMULATED_GPU, "blocking", 1000, 1000)


def managed(runtime, *, clock=lambda: 0, reserve: int = 0) -> ManagedWorker:
    worker = ManagedWorker(
        "worker",
        runtime,
        InMemoryWorkerRegistry(clock=clock),
        memory_safety_reserve_bytes=reserve,
        clock=clock,
    )
    worker.register()
    return worker


def test_duplicate_concurrent_loads_are_coalesced_and_reserved_once():
    definition = model("model", 200)
    runtime = BlockingRuntime(definition)
    worker = managed(runtime)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(worker.load_model, definition)
        assert runtime.started.wait(1)
        second = pool.submit(worker.load_model, definition)
        deadline = time.monotonic() + 1
        while worker.model_cache_metrics().coalesced_loads < 1:
            assert time.monotonic() < deadline
        assert worker.model_cache_metrics().reserved_memory_bytes == 200
        runtime.release.set()
        first.result(timeout=1)
        second.result(timeout=1)

    metrics = worker.model_cache_metrics()
    assert runtime.load_calls == 1
    assert metrics.cold_starts == 1
    assert metrics.cache_misses == 1
    assert metrics.cache_hits == 1
    assert metrics.reserved_memory_bytes == 0


def test_coalesced_load_failure_reaches_all_waiters_and_releases_reservation():
    definition = model("model", 200)
    runtime = BlockingRuntime(definition, fail=True)
    worker = managed(runtime)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(worker.load_model, definition)
        assert runtime.started.wait(1)
        second = pool.submit(worker.load_model, definition)
        deadline = time.monotonic() + 1
        while worker.model_cache_metrics().coalesced_loads < 1:
            assert time.monotonic() < deadline
        runtime.release.set()
        with pytest.raises(RuntimeError, match="load failed"):
            first.result(timeout=1)
        with pytest.raises(RuntimeError, match="coalesced model load failed"):
            second.result(timeout=1)

    metrics = worker.model_cache_metrics()
    assert runtime.load_calls == 1
    assert metrics.load_failures == 1
    assert metrics.reserved_memory_bytes == 0


def test_loading_uses_lru_eviction_and_protects_queued_models():
    first, second = model("first"), model("second")
    runtime = SimulatedGpuRuntime(
        [first, second], SimulatedGpuConfig(total_memory_bytes=250), sleeper=lambda _: None
    )
    now = [0.0]
    worker = managed(runtime, clock=lambda: now[0])
    worker.load_model(first)
    now[0] = 1
    worker.load_model(second)
    assert not runtime.is_model_loaded("first")
    assert runtime.is_model_loaded("second")
    assert worker.model_cache_metrics().evictions == 1

    queued = RequestRecord(
        "request", "tenant", "second", "hello", 1, 1, 0, 100, False
    )
    queued.transition(RequestState.VALIDATED)
    queued.transition(RequestState.ADMITTED)
    queued.transition(RequestState.ASSIGNED)
    worker.enqueue_request(queued)
    with pytest.raises(MemoryError, match="insufficient worker memory"):
        worker.load_model(first)


def test_idle_eviction_uses_model_timeout_and_never_evicts_queued_model():
    definition = model("model", memory=100, idle=5)
    runtime = SimulatedGpuRuntime(
        [definition], SimulatedGpuConfig(total_memory_bytes=1000), sleeper=lambda _: None
    )
    now = [0.0]
    worker = managed(runtime, clock=lambda: now[0])
    worker.load_model(definition)
    now[0] = 6
    assert worker.evict_idle_models() == ("model",)
    assert worker.model_cache_metrics().evictions == 1


def test_memory_pressure_never_evicts_an_actively_generating_model():
    first, second = model("first"), model("second")
    generation_started = Event()
    release_generation = Event()

    def sleeper(seconds: float) -> None:
        if seconds > 0:
            generation_started.set()
            assert release_generation.wait(1)

    runtime = SimulatedGpuRuntime(
        [first, second],
        SimulatedGpuConfig(total_memory_bytes=250, tokens_per_second=100),
        sleeper=sleeper,
    )
    worker = managed(runtime)
    worker.load_model(first)
    item = RequestRecord(
        "request", "tenant", "first", "hello", 1, 1, 0, 100, False
    )
    item.transition(RequestState.VALIDATED)
    item.transition(RequestState.ADMITTED)
    item.transition(RequestState.ASSIGNED)
    worker.enqueue_request(item)

    with ThreadPoolExecutor(max_workers=1) as pool:
        execution = pool.submit(worker.execute_batch)
        assert generation_started.wait(1)
        with pytest.raises(MemoryError, match="insufficient worker memory"):
            worker.load_model(second)
        release_generation.set()
        execution.result(timeout=1)

    assert runtime.is_model_loaded("first")
    assert not runtime.is_model_loaded("second")
    assert item.status == RequestState.COMPLETED


def test_active_cancellation_discards_uninterruptible_runtime_output():
    definition = model("model")
    generation_started = Event()
    release_generation = Event()

    def sleeper(seconds: float) -> None:
        if seconds > 0:
            generation_started.set()
            assert release_generation.wait(1)

    runtime = SimulatedGpuRuntime(
        [definition], SimulatedGpuConfig(total_memory_bytes=1000), sleeper=sleeper
    )
    worker = managed(runtime)
    worker.load_model(definition)
    item = RequestRecord(
        "request", "tenant", "model", "hello", 1, 1, 0, 100, False
    )
    item.transition(RequestState.VALIDATED)
    item.transition(RequestState.ADMITTED)
    item.transition(RequestState.ASSIGNED)
    worker.enqueue_request(item)

    with ThreadPoolExecutor(max_workers=1) as pool:
        execution = pool.submit(worker.execute_batch)
        assert generation_started.wait(1)
        assert worker.cancel_request(item.request_id)
        release_generation.set()
        assert execution.result(timeout=1) == ()
    assert item.status == RequestState.CANCELLED
