from __future__ import annotations

import gc
import random
import tracemalloc
from dataclasses import asdict, dataclass

from inference.kv_cache import KVBlockAllocator, KVCacheCapacityError
from inference.prefix_cache import PrefixCache
from runtime.fault_injection import FaultInjectingRuntime, FaultRule, InjectedRuntimeFault
from runtime.simulated_gpu import SimulatedGpuConfig, SimulatedGpuRuntime
from serving_platform.admission import TenantAdmissionController
from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    RuntimeType,
    TenantLimits,
    WorkerState,
)


@dataclass(frozen=True)
class QualificationResult:
    classification: str
    iterations: int
    kv_invariants_passed: bool
    kv_final_used_blocks: int
    prefix_bounds_passed: bool
    prefix_final_entries: int
    overload_rejection_code: str
    admission_final_reserved_tokens: int
    injected_failure_observed: bool
    post_failure_recovery_passed: bool
    python_heap_growth_bytes: int
    max_python_heap_growth_bytes: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _definition() -> ModelDefinition:
    return ModelDefinition(
        "qualification-model",
        "deterministic-v1",
        RuntimeType.SIMULATED,
        100,
        ("float16",),
        "float16",
        64,
        256,
        True,
        True,
        1,
        60,
    )


def _worker() -> WorkerState:
    return WorkerState(
        "qualification-worker",
        DeviceType.SIMULATED_GPU,
        "deterministic-simulated-gpu",
        1000,
        900,
        resident_models={"qualification-model"},
        health_status=HealthStatus.HEALTHY,
    )


def _request(index: int) -> RequestRecord:
    return RequestRecord(
        f"qualification-{index}",
        "tenant",
        "qualification-model",
        "bounded local request",
        3,
        2,
        0,
        float("inf"),
        False,
    )


def run_qualification(
    iterations: int = 2_000,
    max_python_heap_growth_bytes: int = 2_000_000,
) -> QualificationResult:
    if iterations <= 0 or max_python_heap_growth_bytes < 0:
        raise ValueError("qualification limits are invalid")
    randomizer = random.Random(20260828)
    allocator = KVBlockAllocator(128, 4)
    owners: set[str] = set()
    prefix_cache = PrefixCache(max_entries=16, max_tokens=128)

    gc.collect()
    tracemalloc.start()
    baseline_bytes = tracemalloc.get_traced_memory()[0]
    for index in range(iterations):
        owner = f"owner-{randomizer.randrange(64)}"
        if owner in owners and randomizer.random() < 0.55:
            allocator.release(owner)
            owners.remove(owner)
        elif owner not in owners:
            try:
                allocator.allocate(owner, randomizer.randrange(1, 33))
                owners.add(owner)
            except KVCacheCapacityError:
                pass
        prefix = tuple(range(index % 24 + 1))
        prefix_cache.put(prefix, index)
        allocator.assert_invariants()
        snapshot = prefix_cache.snapshot()
        if snapshot.entries > 16 or snapshot.cached_tokens > 128:
            raise AssertionError("prefix cache exceeded its configured bounds")
    for owner in tuple(owners):
        allocator.release(owner)
    prefix_cache.clear()
    gc.collect()
    heap_growth = max(0, tracemalloc.get_traced_memory()[0] - baseline_bytes)
    tracemalloc.stop()

    definition = _definition()
    admission = TenantAdmissionController(
        [TenantLimits("tenant", 2, 4, 100)],
        global_queue_capacity=4,
    )
    admitted = []
    for index in range(4):
        item = _request(index)
        decision = admission.decide(item, definition, [_worker()])
        if not decision.admitted:
            raise AssertionError("qualification request was rejected before capacity")
        admitted.append(item.request_id)
    overloaded = admission.decide(_request(5), definition, [_worker()])
    for request_id in admitted:
        admission.release(request_id)
    admission_snapshot = admission.snapshot("tenant")

    base_runtime = SimulatedGpuRuntime(
        [definition],
        SimulatedGpuConfig(total_memory_bytes=1000),
        sleeper=lambda _seconds: None,
    )
    runtime = FaultInjectingRuntime(
        base_runtime,
        [FaultRule("generate", fail_on_calls=(1,), code="qualification_fault")],
    )
    runtime.load_model(definition)
    failure_observed = False
    try:
        runtime.generate(definition.model_id, ["hello"], 1, 0)
    except InjectedRuntimeFault:
        failure_observed = True
    recovered = bool(runtime.generate(definition.model_id, ["hello"], 1, 0))

    allocator_snapshot = allocator.snapshot()
    prefix_snapshot = prefix_cache.snapshot()
    passed = all(
        (
            allocator_snapshot.used_blocks == 0,
            prefix_snapshot.entries == 0,
            overloaded.code in {"global_queue_full", "tenant_queue_full"},
            admission_snapshot["reserved_tokens"] == 0,
            failure_observed,
            recovered,
            heap_growth <= max_python_heap_growth_bytes,
        )
    )
    return QualificationResult(
        "measured_local_python_and_deterministic_simulation_not_gpu_performance",
        iterations,
        True,
        allocator_snapshot.used_blocks,
        True,
        prefix_snapshot.entries,
        overloaded.code,
        admission_snapshot["reserved_tokens"],
        failure_observed,
        recovered,
        heap_growth,
        max_python_heap_growth_bytes,
        passed,
    )
