from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from threading import Lock

from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    RuntimeType,
    WorkerState,
)
from serving_platform.registry.interfaces import ModelRegistry
from serving_platform.routing.interfaces import RoutingDecision


class NoEligibleWorker(RuntimeError):
    def __init__(self, rejected: dict[str, str]):
        super().__init__("no eligible worker")
        self.rejected = rejected


class UnknownModel(ValueError):
    pass


def filter_workers(
    request: RequestRecord,
    workers: Sequence[WorkerState],
) -> tuple[tuple[WorkerState, ...], dict[str, str]]:
    eligible: list[WorkerState] = []
    rejected: dict[str, str] = {}
    for worker in sorted(workers, key=lambda item: item.worker_id):
        reason: str | None = None
        if worker.health_status != HealthStatus.HEALTHY:
            reason = f"health:{worker.health_status.value}"
        elif worker.draining:
            reason = "worker_draining"
        elif request.model_id not in worker.resident_models:
            reason = "model_not_resident"
        elif worker.active_batch_count >= worker.max_concurrency:
            reason = "worker_concurrency_saturated"
        if reason:
            rejected[worker.worker_id] = reason
        else:
            eligible.append(worker)
    return tuple(eligible), rejected


class RoundRobinPolicy:
    name = "round_robin"

    def __init__(self) -> None:
        self._next = 0
        self._lock = Lock()

    def select(
        self,
        request: RequestRecord,
        workers: Sequence[WorkerState],
    ) -> RoutingDecision:
        eligible, rejected = filter_workers(request, workers)
        if not eligible:
            raise NoEligibleWorker(rejected)
        with self._lock:
            selected = eligible[self._next % len(eligible)]
            self._next += 1
        return RoutingDecision(
            selected_worker_id=selected.worker_id,
            policy=self.name,
            candidates=tuple(worker.worker_id for worker in eligible),
            rejected=rejected,
            scores={worker.worker_id: 0.0 for worker in eligible},
            scoring_inputs={worker.worker_id: {} for worker in eligible},
        )


class LeastQueueDepthPolicy:
    name = "least_queue_depth"

    def select(
        self,
        request: RequestRecord,
        workers: Sequence[WorkerState],
    ) -> RoutingDecision:
        eligible, rejected = filter_workers(request, workers)
        if not eligible:
            raise NoEligibleWorker(rejected)
        scores = {worker.worker_id: float(worker.queue_depth) for worker in eligible}
        selected = min(eligible, key=lambda worker: (scores[worker.worker_id], worker.worker_id))
        return RoutingDecision(
            selected_worker_id=selected.worker_id,
            policy=self.name,
            candidates=tuple(worker.worker_id for worker in eligible),
            rejected=rejected,
            scores=scores,
            scoring_inputs={
                worker.worker_id: {"queue_depth": worker.queue_depth}
                for worker in eligible
            },
        )


class PlacementPolicy:
    def __init__(
        self,
        models: ModelRegistry,
        memory_safety_reserve_bytes: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if memory_safety_reserve_bytes < 0:
            raise ValueError("memory safety reserve cannot be negative")
        self.models = models
        self.memory_safety_reserve_bytes = memory_safety_reserve_bytes
        self._clock = clock

    def _model(self, model_id: str) -> ModelDefinition:
        model = self.models.get(model_id)
        if model is None:
            raise UnknownModel(f"model is not registered: {model_id}")
        return model

    def _filter(
        self,
        request: RequestRecord,
        workers: Sequence[WorkerState],
    ) -> tuple[ModelDefinition, tuple[WorkerState, ...], dict[str, str]]:
        model = self._model(request.model_id)
        eligible: list[WorkerState] = []
        rejected: dict[str, str] = {}
        for worker in sorted(workers, key=lambda item: item.worker_id):
            reason: str | None = None
            resident = model.model_id in worker.resident_models
            runtime_compatible = (
                model.runtime_type == RuntimeType.SIMULATED
                and worker.device_type == DeviceType.SIMULATED_GPU
            ) or (
                model.runtime_type == RuntimeType.HUGGINGFACE
                and worker.device_type in {DeviceType.CPU, DeviceType.CUDA}
            )
            if worker.health_status != HealthStatus.HEALTHY:
                reason = f"health:{worker.health_status.value}"
            elif worker.draining:
                reason = "worker_draining"
            elif worker.active_batch_count >= worker.max_concurrency:
                reason = "worker_concurrency_saturated"
            elif not runtime_compatible:
                reason = "runtime_incompatible"
            elif request.estimated_tokens > model.max_context_tokens:
                reason = "context_window_exceeded"
            elif not resident:
                if worker.available_memory_bytes is None:
                    reason = "memory_unknown_for_cold_load"
                elif (
                    worker.available_memory_bytes - self.memory_safety_reserve_bytes
                    < model.estimated_memory_bytes
                ):
                    reason = "insufficient_memory_after_safety_reserve"
            if reason:
                rejected[worker.worker_id] = reason
            else:
                eligible.append(worker)
        return model, tuple(eligible), rejected

    @staticmethod
    def _decision(
        selected: WorkerState,
        policy: str,
        eligible: Sequence[WorkerState],
        rejected: dict[str, str],
        scores: dict[str, float],
        inputs: dict[str, dict[str, float | int | bool | None]],
    ) -> RoutingDecision:
        return RoutingDecision(
            selected.worker_id,
            policy,
            tuple(worker.worker_id for worker in eligible),
            rejected,
            scores,
            inputs,
        )


class ModelResidencyAwarePolicy(PlacementPolicy):
    name = "model_residency_aware"

    def select(
        self, request: RequestRecord, workers: Sequence[WorkerState]
    ) -> RoutingDecision:
        model, eligible, rejected = self._filter(request, workers)
        if not eligible:
            raise NoEligibleWorker(rejected)
        scores = {
            worker.worker_id: float(model.model_id not in worker.resident_models)
            for worker in eligible
        }
        inputs: dict[str, dict[str, float | int | bool | None]] = {
            worker.worker_id: {
                "model_resident": model.model_id in worker.resident_models,
                "queue_depth": worker.queue_depth,
                "available_memory_bytes": worker.available_memory_bytes,
            }
            for worker in eligible
        }
        selected = min(
            eligible,
            key=lambda worker: (scores[worker.worker_id], worker.queue_depth, worker.worker_id),
        )
        return self._decision(selected, self.name, eligible, rejected, scores, inputs)


class MemoryAwareLeastLoadedPolicy(PlacementPolicy):
    name = "memory_aware_least_loaded"

    def select(
        self, request: RequestRecord, workers: Sequence[WorkerState]
    ) -> RoutingDecision:
        model, eligible, rejected = self._filter(request, workers)
        if not eligible:
            raise NoEligibleWorker(rejected)
        scores: dict[str, float] = {}
        inputs: dict[str, dict[str, float | int | bool | None]] = {}
        for worker in eligible:
            resident = model.model_id in worker.resident_models
            projected_model_bytes = 0 if resident else model.estimated_memory_bytes
            if worker.total_memory_bytes and worker.available_memory_bytes is not None:
                projected_used = (
                    worker.total_memory_bytes
                    - worker.available_memory_bytes
                    + projected_model_bytes
                    + self.memory_safety_reserve_bytes
                )
                memory_utilization = min(projected_used / worker.total_memory_bytes, 1.0)
            else:
                memory_utilization = 0.5
            concurrency_utilization = worker.active_batch_count / worker.max_concurrency
            score = memory_utilization + concurrency_utilization + worker.queue_depth
            scores[worker.worker_id] = score
            inputs[worker.worker_id] = {
                "model_resident": resident,
                "available_memory_bytes": worker.available_memory_bytes,
                "projected_model_bytes": projected_model_bytes,
                "memory_safety_reserve_bytes": self.memory_safety_reserve_bytes,
                "memory_utilization": memory_utilization,
                "concurrency_utilization": concurrency_utilization,
                "queue_depth": worker.queue_depth,
            }
        selected = min(eligible, key=lambda worker: (scores[worker.worker_id], worker.worker_id))
        return self._decision(selected, self.name, eligible, rejected, scores, inputs)


class EstimatedCompletionTimePolicy(PlacementPolicy):
    name = "estimated_completion_time"

    def __init__(
        self,
        models: ModelRegistry,
        memory_safety_reserve_bytes: int = 0,
        fallback_tokens_per_second: float = 1,
        batching_penalty_ratio: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(models, memory_safety_reserve_bytes, clock)
        if fallback_tokens_per_second <= 0 or batching_penalty_ratio < 0:
            raise ValueError("completion estimate parameters are invalid")
        self.fallback_tokens_per_second = fallback_tokens_per_second
        self.batching_penalty_ratio = batching_penalty_ratio

    def select(
        self, request: RequestRecord, workers: Sequence[WorkerState]
    ) -> RoutingDecision:
        model, initially_eligible, rejected = self._filter(request, workers)
        eligible: list[WorkerState] = []
        scores: dict[str, float] = {}
        inputs: dict[str, dict[str, float | int | bool | None]] = {}
        for worker in initially_eligible:
            throughput = worker.recent_tokens_per_second or self.fallback_tokens_per_second
            generation_seconds = request.estimated_tokens / throughput
            queue_delay_seconds = (
                worker.queue_depth * request.estimated_tokens
                / (throughput * worker.max_concurrency)
            )
            model_load_seconds = (
                0.0 if model.model_id in worker.resident_models else model.load_timeout_seconds
            )
            batching_penalty_seconds = (
                generation_seconds
                * self.batching_penalty_ratio
                * worker.active_batch_count
                / worker.max_concurrency
            )
            estimated_seconds = (
                queue_delay_seconds
                + model_load_seconds
                + generation_seconds
                + batching_penalty_seconds
            )
            if self._clock() + estimated_seconds > request.deadline:
                rejected[worker.worker_id] = "deadline_not_feasible"
                continue
            eligible.append(worker)
            scores[worker.worker_id] = estimated_seconds
            inputs[worker.worker_id] = {
                "estimated_queue_delay_seconds": queue_delay_seconds,
                "model_load_penalty_seconds": model_load_seconds,
                "estimated_generation_seconds": generation_seconds,
                "batching_penalty_seconds": batching_penalty_seconds,
                "recent_tokens_per_second": worker.recent_tokens_per_second,
                "fallback_throughput_used": worker.recent_tokens_per_second <= 0,
            }
        if not eligible:
            raise NoEligibleWorker(rejected)
        selected = min(eligible, key=lambda worker: (scores[worker.worker_id], worker.worker_id))
        return self._decision(selected, self.name, eligible, rejected, scores, inputs)
