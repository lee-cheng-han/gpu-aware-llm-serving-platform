from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from apps.control_plane.scheduler import GlobalScheduler, RequestDeadlineExceeded
from apps.control_plane.service import ControlPlane, WorkerDirectory
from apps.worker import ManagedWorker
from runtime.base import RuntimeResult
from runtime.simulated_gpu import SimulatedGpuConfig, SimulatedGpuRuntime
from serving_platform.admission import TenantAdmissionController
from serving_platform.domain import (
    ModelDefinition,
    RequestRecord,
    RequestState,
    RuntimeType,
    TenantLimits,
)
from serving_platform.registry import InMemoryModelRegistry, InMemoryWorkerRegistry
from serving_platform.request_state.interfaces import RequestStateStore
from serving_platform.routing import ModelResidencyAwarePolicy, NoEligibleWorker
from serving_platform.scheduling import WeightedFairRequestQueue


class PlatformAdmissionError(RuntimeError):
    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code = code


class PlatformExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlatformGenerationResult:
    request: RequestRecord
    result: RuntimeResult
    scheduler_policy: str
    batch_size: int


class LocalControlPlanePipeline:
    """Async gateway adapter over the complete in-process control-plane path."""

    def __init__(
        self,
        model: ModelDefinition,
        registry: InMemoryWorkerRegistry,
        models: InMemoryModelRegistry,
        requests: RequestStateStore,
        admission: TenantAdmissionController,
        fair_queue: WeightedFairRequestQueue,
        control_plane: ControlPlane,
        workers: WorkerDirectory,
    ):
        self.model = model
        self.registry = registry
        self.models = models
        self.requests = requests
        self.admission = admission
        self.fair_queue = fair_queue
        self.control_plane = control_plane
        self.workers = workers
        self._execution_lock = asyncio.Lock()

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(prompt.split())

    async def submit(
        self,
        tenant_id: str,
        prompt: str,
        prompt_tokens: int,
        max_new_tokens: int,
        temperature: float,
        priority: int,
        deadline_seconds: float,
    ) -> PlatformGenerationResult:
        async with self._execution_lock:
            now = time.monotonic()
            request = RequestRecord(
                request_id=f"platform_{uuid.uuid4().hex}",
                tenant_id=tenant_id,
                model_id=self.model.model_id,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                priority=priority,
                deadline=now + deadline_seconds,
                stream=False,
                temperature=temperature,
                created_at=now,
            )
            self.requests.create(request)
            request.transition(RequestState.VALIDATED, now)
            self.requests.save(request)
            decision = self.admission.decide(request, self.model, self.registry.list())
            if not decision.admitted:
                request.transition(RequestState.REJECTED)
                self.requests.save(request)
                raise PlatformAdmissionError(decision.code, decision.reason)
            request.transition(RequestState.ADMITTED)
            self.fair_queue.enqueue(request)
            self.requests.save(request)
            selected = None
            while selected is None:
                selected = self.fair_queue.pop()
            if selected.request_id != request.request_id:
                request.transition(RequestState.FAILED)
                self.requests.save(request)
                self.admission.release(request.request_id)
                raise PlatformExecutionError("control-plane queue ownership was violated")
            # This opt-in path uses only the bounded deterministic simulator, whose maximum
            # configured delay is small. Real model workers remain off the event loop and
            # will move behind the process transport in the next phase.
            return self._execute(selected)

    def _execute(self, request: RequestRecord) -> PlatformGenerationResult:
        try:
            assignment = self.control_plane.dispatch(request)
            self.admission.mark_running(request.request_id)
            worker = self.workers.get(assignment.worker_id)
            if worker is None:
                raise PlatformExecutionError("assigned worker became unavailable")
            batch = worker.execute_batch()
            execution = next(
                (item for item in batch if item.request_id == request.request_id), None
            )
            if execution is None:
                raise PlatformExecutionError("worker did not return the assigned request")
            self.requests.save(request)
            return PlatformGenerationResult(request, execution.result, assignment.policy, len(batch))
        except (NoEligibleWorker, RequestDeadlineExceeded) as exc:
            if not request.terminal:
                request.transition(
                    RequestState.TIMED_OUT
                    if request.deadline <= time.monotonic()
                    else RequestState.FAILED
                )
                self.requests.save(request)
            raise PlatformExecutionError(str(exc)) from exc
        except BaseException as exc:
            if not request.terminal:
                request.transition(RequestState.FAILED)
                self.requests.save(request)
            if isinstance(exc, PlatformExecutionError):
                raise
            raise PlatformExecutionError("control-plane execution failed") from exc
        finally:
            self.admission.release(request.request_id)


def build_local_simulated_pipeline(
    requests: RequestStateStore,
    tenant_ids: set[str],
    max_queue_size: int,
    max_concurrent_requests: int,
    max_context_tokens: int,
    max_batch_tokens: int,
) -> LocalControlPlanePipeline:
    effective_context_tokens = min(max_context_tokens, max_batch_tokens)
    model = ModelDefinition(
        model_id="local-simulated-model",
        revision="deterministic-v1",
        runtime_type=RuntimeType.SIMULATED,
        estimated_memory_bytes=256 * 1024**2,
        supported_dtypes=("float16",),
        default_dtype="float16",
        max_context_tokens=effective_context_tokens,
        max_batch_tokens=max_batch_tokens,
        supports_streaming=False,
        supports_cancellation=True,
        load_timeout_seconds=5,
        idle_eviction_seconds=300,
    )
    models = InMemoryModelRegistry([model])
    registry = InMemoryWorkerRegistry()
    directory = WorkerDirectory()
    for index, throughput in enumerate((10_000.0, 8_000.0), start=1):
        runtime = SimulatedGpuRuntime(
            [model],
            SimulatedGpuConfig(
                total_memory_bytes=4 * 1024**3,
                tokens_per_second=throughput,
                seed=index,
                device_name=f"local-simulated-gpu-{index}",
            ),
        )
        worker = ManagedWorker(
            f"local-sim-worker-{index}",
            runtime,
            registry,
            max_queue_depth=max_queue_size,
            max_concurrency=max_concurrent_requests,
            max_batch_tokens=max_batch_tokens,
        )
        worker.register()
        directory.add(worker)
    tenants = [
        TenantLimits(
            tenant_id,
            max_concurrent_requests,
            max_queue_size,
            effective_context_tokens * max_queue_size,
        )
        for tenant_id in sorted(tenant_ids)
    ]
    admission = TenantAdmissionController(tenants, max_queue_size)
    fair_queue = WeightedFairRequestQueue(tenants)
    control_plane = ControlPlane(
        GlobalScheduler(registry, ModelResidencyAwarePolicy(models)),
        directory,
        models,
        requests,
        fair_queue,
    )
    return LocalControlPlanePipeline(
        model,
        registry,
        models,
        requests,
        admission,
        fair_queue,
        control_plane,
        directory,
    )
