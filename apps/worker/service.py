from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from threading import Condition, RLock
from typing import Protocol

from runtime.base import ModelRuntime, RuntimeCapacity, RuntimeResult
from runtime.huggingface import HuggingFaceRuntime
from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    RequestState,
    WorkerState,
)
from serving_platform.registry.interfaces import WorkerRegistry


class Worker(Protocol):
    def register(self) -> WorkerState: ...
    def heartbeat(self) -> WorkerState: ...
    def load_model(self, model: ModelDefinition) -> None: ...
    def unload_model(self, model_id: str) -> None: ...
    def warmup_model(self, model_id: str) -> None: ...
    def enqueue_request(self, request: RequestRecord) -> None: ...
    def cancel_request(self, request_id: str) -> bool: ...
    def execute_batch(self) -> tuple[WorkerExecutionResult, ...]: ...
    def drain(self) -> None: ...
    def shutdown(self) -> None: ...
    def health(self) -> HealthStatus: ...
    def capacity(self) -> RuntimeCapacity: ...


@dataclass(frozen=True)
class WorkerExecutionResult:
    request_id: str
    result: RuntimeResult


@dataclass(frozen=True)
class ModelCacheMetrics:
    cache_hits: int
    cache_misses: int
    cold_starts: int
    load_failures: int
    evictions: int
    reserved_memory_bytes: int
    coalesced_loads: int
    model_load_seconds_total: float


class ManagedWorker:
    """Operational worker shell shared by real and simulated runtimes."""

    def __init__(
        self,
        worker_id: str,
        runtime: ModelRuntime,
        registry: WorkerRegistry,
        max_queue_depth: int = 128,
        max_concurrency: int = 1,
        max_batch_size: int = 8,
        max_batch_tokens: int = 4096,
        memory_safety_reserve_bytes: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if memory_safety_reserve_bytes < 0 or not worker_id or min(
            max_queue_depth, max_concurrency, max_batch_size, max_batch_tokens
        ) <= 0:
            raise ValueError("worker id, queue depth, and concurrency must be positive")
        self.worker_id = worker_id
        self.runtime = runtime
        self.registry = registry
        self.max_queue_depth = max_queue_depth
        self.max_concurrency = max_concurrency
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.memory_safety_reserve_bytes = memory_safety_reserve_bytes
        self._clock = clock
        self._health = HealthStatus.REGISTERING
        self._draining = False
        self._resident_models: set[str] = set()
        self._loading_models: set[str] = set()
        self._model_definitions: dict[str, ModelDefinition] = {}
        self._model_last_used: dict[str, float] = {}
        self._active_requests_by_model: Counter[str] = Counter()
        self._active_requests: dict[str, RequestRecord] = {}
        self._load_attempts: Counter[str] = Counter()
        self._load_failures: dict[str, tuple[int, BaseException]] = {}
        self._reserved_model_memory_bytes = 0
        self._cache_hits = self._cache_misses = self._cold_starts = 0
        self._model_load_failures = self._model_evictions = 0
        self._coalesced_loads = 0
        self._model_load_seconds_total = 0.0
        self._queue: deque[RequestRecord] = deque()
        self._active_batch_count = 0
        self._recent_tokens_per_second = 0.0
        self._lock = RLock()
        self._model_condition = Condition(self._lock)

    def _snapshot(self) -> WorkerState:
        capacity = self.runtime.capacity()
        return WorkerState(
            worker_id=self.worker_id,
            device_type=capacity.device_type,
            device_name=capacity.device_name,
            total_memory_bytes=capacity.total_memory_bytes,
            available_memory_bytes=capacity.available_memory_bytes,
            resident_models=set(self._resident_models),
            loading_models=set(self._loading_models),
            queue_depth=len(self._queue),
            active_batch_count=self._active_batch_count,
            max_concurrency=self.max_concurrency,
            recent_tokens_per_second=self._recent_tokens_per_second,
            health_status=self._health,
            draining=self._draining,
            last_heartbeat=self._clock(),
            allocated_memory_bytes=capacity.allocated_memory_bytes,
            reserved_memory_bytes=capacity.reserved_memory_bytes,
        )

    def register(self) -> WorkerState:
        with self._lock:
            if self._health == HealthStatus.STOPPED:
                self._draining = False
            self._health = HealthStatus.HEALTHY
            snapshot = self._snapshot()
            self.registry.register(snapshot)
            return snapshot

    def heartbeat(self) -> WorkerState:
        with self._lock:
            if self._health != HealthStatus.HEALTHY:
                raise RuntimeError("only a healthy registered worker can heartbeat")
            snapshot = self._snapshot()
            self.registry.heartbeat(snapshot)
            return snapshot

    def load_model(self, model: ModelDefinition) -> None:
        evictions: list[tuple[str, ModelDefinition]] = []
        with self._model_condition:
            if self._health != HealthStatus.HEALTHY or self._draining:
                raise RuntimeError("worker is not accepting model loads")
            if model.model_id in self._resident_models:
                self._cache_hits += 1
                self._model_last_used[model.model_id] = self._clock()
                return
            waiting_for_attempt = self._load_attempts[model.model_id]
            joined_existing_load = model.model_id in self._loading_models
            if joined_existing_load:
                self._coalesced_loads += 1
            while model.model_id in self._loading_models:
                self._model_condition.wait()
            if model.model_id in self._resident_models:
                self._cache_hits += 1
                self._model_last_used[model.model_id] = self._clock()
                return
            failed = self._load_failures.get(model.model_id)
            if (
                joined_existing_load
                and failed is not None
                and failed[0] == waiting_for_attempt
            ):
                raise RuntimeError(f"coalesced model load failed: {model.model_id}") from failed[1]

            capacity = self.runtime.capacity()
            available = capacity.available_memory_bytes
            required = model.estimated_memory_bytes
            if available is not None:
                usable = (
                    available
                    - self._reserved_model_memory_bytes
                    - self.memory_safety_reserve_bytes
                )
                if usable < required:
                    for model_id in sorted(
                        self._resident_models,
                        key=lambda item: (self._model_last_used.get(item, 0), item),
                    ):
                        if self._active_requests_by_model[model_id] or any(
                            request.model_id == model_id for request in self._queue
                        ):
                            continue
                        definition = self._model_definitions[model_id]
                        evictions.append((model_id, definition))
                        usable += definition.estimated_memory_bytes
                        if usable >= required:
                            break
                    if usable < required:
                        raise MemoryError(f"insufficient worker memory for {model.model_id}")
                for model_id, _ in evictions:
                    self._resident_models.remove(model_id)
                    self._model_last_used.pop(model_id, None)

            self._cache_misses += 1
            self._cold_starts += 1
            self._load_attempts[model.model_id] += 1
            attempt = self._load_attempts[model.model_id]
            self._loading_models.add(model.model_id)
            self._reserved_model_memory_bytes += required
        load_started = time.perf_counter()
        try:
            unloaded: list[tuple[str, ModelDefinition]] = []
            for model_id, definition in evictions:
                self.runtime.unload_model(model_id)
                unloaded.append((model_id, definition))
            self.runtime.load_model(model)
            self.runtime.warmup_model(model.model_id)
            with self._model_condition:
                self._model_definitions[model.model_id] = model
                self._resident_models.add(model.model_id)
                self._model_last_used[model.model_id] = self._clock()
                self._model_evictions += len(unloaded)
                self._load_failures.pop(model.model_id, None)
        except BaseException as exc:
            with self._model_condition:
                self._model_load_failures += 1
                self._load_failures[model.model_id] = (attempt, exc)
                for model_id, definition in evictions:
                    if self.runtime.is_model_loaded(model_id):
                        self._resident_models.add(model_id)
                        self._model_definitions[model_id] = definition
                        self._model_last_used[model_id] = self._clock()
            raise
        finally:
            with self._model_condition:
                self._model_load_seconds_total += time.perf_counter() - load_started
                self._loading_models.discard(model.model_id)
                self._reserved_model_memory_bytes -= required
                self._model_condition.notify_all()

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            if any(request.model_id == model_id for request in self._queue):
                raise RuntimeError("cannot unload a model with queued requests")
            if model_id not in self._resident_models:
                return
            self._resident_models.remove(model_id)
            self._model_last_used.pop(model_id, None)
        try:
            self.runtime.unload_model(model_id)
        except BaseException:
            with self._lock:
                self._resident_models.add(model_id)
                self._model_last_used[model_id] = self._clock()
            raise

    def warmup_model(self, model_id: str) -> None:
        with self._lock:
            if model_id not in self._resident_models:
                raise RuntimeError("model must be loaded before warmup")
        self.runtime.warmup_model(model_id)

    def evict_idle_models(self) -> tuple[str, ...]:
        now = self._clock()
        with self._lock:
            eligible = [
                model_id
                for model_id in self._resident_models
                if self._active_requests_by_model[model_id] == 0
                and not any(request.model_id == model_id for request in self._queue)
                and now - self._model_last_used.get(model_id, now)
                >= self._model_definitions[model_id].idle_eviction_seconds
            ]
        evicted: list[str] = []
        for model_id in sorted(eligible, key=lambda item: self._model_last_used[item]):
            self.unload_model(model_id)
            evicted.append(model_id)
        with self._lock:
            self._model_evictions += len(evicted)
        return tuple(evicted)

    def model_cache_metrics(self) -> ModelCacheMetrics:
        with self._lock:
            return ModelCacheMetrics(
                self._cache_hits,
                self._cache_misses,
                self._cold_starts,
                self._model_load_failures,
                self._model_evictions,
                self._reserved_model_memory_bytes,
                self._coalesced_loads,
                self._model_load_seconds_total,
            )

    def enqueue_request(self, request: RequestRecord) -> None:
        with self._lock:
            if self._health != HealthStatus.HEALTHY or self._draining:
                raise RuntimeError("worker is not accepting requests")
            if request.model_id not in self._resident_models:
                raise RuntimeError("request model is not resident")
            if len(self._queue) >= self.max_queue_depth:
                raise OverflowError("worker queue is full")
            if request.estimated_tokens > self.max_batch_tokens:
                raise OverflowError("request exceeds the worker batch token budget")
            if request.status != RequestState.ASSIGNED:
                raise ValueError("only assigned requests can enter a worker queue")
            request.transition(RequestState.QUEUED)
            self._queue.append(request)

    def cancel_request(self, request_id: str) -> bool:
        with self._lock:
            for request in tuple(self._queue):
                if request.request_id == request_id:
                    self._queue.remove(request)
                    request.transition(RequestState.CANCELLED)
                    return True
            active = self._active_requests.get(request_id)
            if active is not None and not active.terminal:
                active.transition(RequestState.CANCELLED)
                return True
        return False

    def execute_batch(self) -> tuple[WorkerExecutionResult, ...]:
        """Execute one compatible local batch; blocking runtime work happens unlocked."""
        with self._lock:
            now = self._clock()
            while self._queue and (
                self._queue[0].terminal or self._queue[0].deadline <= now
            ):
                skipped = self._queue.popleft()
                if not skipped.terminal:
                    skipped.transition(RequestState.TIMED_OUT, now)
            if not self._queue:
                return ()
            first = self._queue.popleft()
            selected = [first]
            deferred: deque[RequestRecord] = deque()
            batch_tokens = first.estimated_tokens
            while self._queue and len(selected) < self.max_batch_size:
                candidate = self._queue.popleft()
                if candidate.terminal:
                    continue
                if candidate.deadline <= now:
                    candidate.transition(RequestState.TIMED_OUT, now)
                    continue
                compatible = (
                    candidate.model_id == first.model_id
                    and candidate.max_new_tokens == first.max_new_tokens
                    and candidate.temperature == first.temperature
                    and candidate.stream == first.stream
                )
                if compatible and batch_tokens + candidate.estimated_tokens <= self.max_batch_tokens:
                    selected.append(candidate)
                    batch_tokens += candidate.estimated_tokens
                else:
                    deferred.append(candidate)
            self._queue.extendleft(reversed(deferred))
            for request in selected:
                request.transition(RequestState.RUNNING, now)
                self._active_requests_by_model[request.model_id] += 1
                self._active_requests[request.request_id] = request
            self._model_last_used[first.model_id] = now
            self._active_batch_count += 1

        started = time.perf_counter()
        try:
            if first.stream:
                raise RuntimeError("streaming requests require the streaming worker path")
            results = self.runtime.generate(
                first.model_id,
                [request.prompt for request in selected],
                first.max_new_tokens,
                first.temperature,
            )
            if len(results) != len(selected):
                raise RuntimeError("runtime returned an unexpected result count")
        except BaseException:
            finished = self._clock()
            with self._lock:
                for request in selected:
                    if not request.terminal:
                        request.transition(RequestState.FAILED, finished)
            raise
        finally:
            with self._lock:
                self._active_batch_count -= 1
                for request in selected:
                    self._active_requests_by_model[request.model_id] -= 1
                    self._active_requests.pop(request.request_id, None)

        elapsed = max(time.perf_counter() - started, 1e-9)
        with self._lock:
            for request in selected:
                if not request.terminal:
                    request.transition(RequestState.COMPLETED, self._clock())
            self._recent_tokens_per_second = sum(
                result.output_tokens for result in results
            ) / elapsed
        return tuple(
            WorkerExecutionResult(request.request_id, result)
            for request, result in zip(selected, results, strict=True)
            if request.status == RequestState.COMPLETED
        )

    def drain(self) -> None:
        with self._lock:
            if self._health == HealthStatus.STOPPED:
                raise RuntimeError("stopped worker cannot drain")
            self._draining = True

    def shutdown(self) -> None:
        with self._lock:
            self._draining = True
            while self._queue:
                self._queue.popleft().transition(RequestState.FAILED)
            resident_models = tuple(self._resident_models)
            self._resident_models.clear()
            self._health = HealthStatus.STOPPED
        try:
            for model_id in resident_models:
                self.runtime.unload_model(model_id)
        finally:
            self.registry.unregister(self.worker_id)

    def health(self) -> HealthStatus:
        return self._health

    def capacity(self) -> RuntimeCapacity:
        return self.runtime.capacity()


class CpuHuggingFaceWorker(ManagedWorker):
    def __init__(self, worker_id: str, model: ModelDefinition, registry: WorkerRegistry):
        super().__init__(worker_id, HuggingFaceRuntime(model), registry)


class CudaHuggingFaceWorker(ManagedWorker):
    def __init__(
        self,
        worker_id: str,
        model: ModelDefinition,
        registry: WorkerRegistry,
        device_index: int = 0,
    ):
        runtime = HuggingFaceRuntime(
            model, device_type=DeviceType.CUDA, cuda_device_index=device_index
        )
        # Fail at construction rather than silently presenting a fake CUDA worker.
        runtime.capacity()
        super().__init__(worker_id, runtime, registry)


def create_huggingface_worker(
    worker_id: str,
    model: ModelDefinition,
    registry: WorkerRegistry,
) -> ManagedWorker:
    """Use CUDA automatically when PyTorch reports it available, otherwise CPU."""
    torch = import_module("torch")
    if torch.cuda.is_available():
        return CudaHuggingFaceWorker(worker_id, model, registry)
    return CpuHuggingFaceWorker(worker_id, model, registry)
