"""Deterministic control-plane evaluation; outputs are simulations, not GPU measurements."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from serving_platform.domain import RequestRecord, RequestState, TenantLimits
from serving_platform.scheduling import WeightedFairRequestQueue


@dataclass
class SimulatedWorker:
    worker_id: str
    tokens_per_second: float
    max_resident_models: int
    resident_models: list[str]
    available_at: float = 0
    last_used: dict[str, int] = field(default_factory=dict)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def _expand_requests(document: dict[str, Any]) -> list[RequestRecord]:
    requests: list[RequestRecord] = []
    sequence = 0
    for group in document["request_groups"]:
        tenant_ids = group["tenants"]
        model_ids = group["models"]
        priorities = group.get("priorities", [0])
        for offset in range(group["count"]):
            arrival = float(group.get("arrival_seconds", 0))
            request = RequestRecord(
                request_id=f"request-{sequence:04d}",
                tenant_id=tenant_ids[offset % len(tenant_ids)],
                model_id=model_ids[offset % len(model_ids)],
                prompt=f"synthetic evaluation request {sequence}",
                prompt_tokens=int(group["prompt_tokens"]),
                max_new_tokens=int(group["output_tokens"]),
                priority=int(priorities[offset % len(priorities)]),
                deadline=arrival + float(group["deadline_seconds"]),
                stream=False,
                created_at=arrival,
            )
            request.transition(RequestState.VALIDATED, arrival)
            request.transition(RequestState.ADMITTED, arrival)
            requests.append(request)
            sequence += 1
    return requests


def load_workload(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    required = {"name", "seed", "tenants", "models", "workers", "request_groups", "gates"}
    missing = required - document.keys()
    if missing:
        raise ValueError(f"workload is missing fields: {', '.join(sorted(missing))}")
    if not document["tenants"] or not document["models"] or not document["workers"]:
        raise ValueError("workload tenants, models, and workers cannot be empty")
    return document


def _fairness(completed: dict[str, int], weights: dict[str, int]) -> float:
    normalized = [completed.get(tenant, 0) / weight for tenant, weight in weights.items()]
    denominator = len(normalized) * sum(value * value for value in normalized)
    return (sum(normalized) ** 2 / denominator) if denominator else 1


def simulate(document: dict[str, Any]) -> dict[str, Any]:
    tenant_weights = {item["tenant_id"]: int(item["weight"]) for item in document["tenants"]}
    limits = [
        TenantLimits(tenant_id, 10_000, 10_000, 10_000_000, weight)
        for tenant_id, weight in tenant_weights.items()
    ]
    queue = WeightedFairRequestQueue(
        limits,
        base_quantum_tokens=int(document.get("fair_queue_quantum_tokens", 128)),
        priority_aging_seconds=float(document.get("priority_aging_seconds", 1)),
        clock=lambda: 0,
    )
    requests = _expand_requests(document)
    for request in requests:
        queue.enqueue(request)

    workers = [
        SimulatedWorker(
            item["worker_id"],
            float(item["tokens_per_second"]),
            int(item["max_resident_models"]),
            list(item.get("resident_models", [])),
        )
        for item in document["workers"]
    ]
    models = document["models"]
    latencies: list[float] = []
    decision_times: list[float] = []
    completed: dict[str, int] = {tenant: 0 for tenant in tenant_weights}
    scheduled: dict[str, int] = {tenant: 0 for tenant in tenant_weights}
    deadline_misses = cache_hits = cache_misses = total_output_tokens = tick = 0
    service_order: list[str] = []

    while queue.depth():
        decision_started = time.perf_counter_ns()
        request = queue.pop()
        if request is None:
            continue
        choices: list[tuple[float, str, SimulatedWorker, bool]] = []
        for worker in workers:
            hit = request.model_id in worker.resident_models
            load = 0 if hit else float(models[request.model_id]["load_seconds"])
            finish = max(worker.available_at, request.created_at) + load + (
                request.estimated_tokens / worker.tokens_per_second
            )
            choices.append((finish, worker.worker_id, worker, hit))
        finish, _, worker, hit = min(choices)
        decision_times.append((time.perf_counter_ns() - decision_started) / 1000)

        if hit:
            cache_hits += 1
        else:
            cache_misses += 1
            if len(worker.resident_models) >= worker.max_resident_models:
                victim = min(worker.resident_models, key=lambda model: worker.last_used.get(model, -1))
                worker.resident_models.remove(victim)
            worker.resident_models.append(request.model_id)
        tick += 1
        worker.last_used[request.model_id] = tick
        worker.available_at = finish
        scheduled[request.tenant_id] += 1
        service_order.append(request.tenant_id)
        latency_ms = (finish - request.created_at) * 1000
        latencies.append(latency_ms)
        if finish > request.deadline:
            deadline_misses += 1
        else:
            completed[request.tenant_id] += 1
            total_output_tokens += request.max_new_tokens

    makespan = max((worker.available_at for worker in workers), default=0)
    total = len(requests)
    return {
        "classification": "deterministic_simulation_not_real_hardware",
        "workload": document["name"],
        "seed": document["seed"],
        "request_count": total,
        "metrics": {
            "completed_requests": total - deadline_misses,
            "deadline_misses": deadline_misses,
            "deadline_miss_ratio": deadline_misses / total if total else 0,
            "simulated_tokens_per_second": total_output_tokens / makespan if makespan else 0,
            "p50_latency_ms": statistics.median(latencies) if latencies else 0,
            "p95_latency_ms": percentile(latencies, 0.95),
            "scheduler_decision_p95_us": percentile(decision_times, 0.95),
            "cache_hit_rate": cache_hits / (cache_hits + cache_misses),
            "weighted_fairness": _fairness(scheduled, tenant_weights),
            "tenant_scheduled": scheduled,
            "tenant_completed": completed,
            "makespan_seconds": makespan,
        },
        "service_order": service_order,
        "gates": document["gates"],
    }


def check_gates(result: dict[str, Any]) -> list[str]:
    metrics = result["metrics"]
    failures: list[str] = []
    for name, expected in result["gates"].items():
        direction, metric = name.split("_", 1)
        actual = metrics[metric]
        failed = actual < expected if direction == "min" else actual > expected
        if failed:
            failures.append(f"{metric}: expected {direction} {expected}, observed {actual}")
    return failures


def render_markdown(result: dict[str, Any], failures: list[str]) -> str:
    metrics = result["metrics"]
    rows = [
        ("Completed requests", metrics["completed_requests"]),
        ("Deadline miss ratio", f'{metrics["deadline_miss_ratio"]:.4f}'),
        ("Simulated tokens/s", f'{metrics["simulated_tokens_per_second"]:.2f}'),
        ("p50 latency (ms)", f'{metrics["p50_latency_ms"]:.2f}'),
        ("p95 latency (ms)", f'{metrics["p95_latency_ms"]:.2f}'),
        ("Scheduler decision p95 (us)", f'{metrics["scheduler_decision_p95_us"]:.2f}'),
        ("Cache hit rate", f'{metrics["cache_hit_rate"]:.4f}'),
        ("Weighted fairness", f'{metrics["weighted_fairness"]:.4f}'),
    ]
    body = "\n".join(f"| {name} | {value} |" for name, value in rows)
    status = "PASS" if not failures else "FAIL"
    return (
        f"# Simulated evaluation: {result['workload']}\n\n"
        "> This is a deterministic control-plane simulation, not a real GPU benchmark.\n\n"
        f"Regression gates: **{status}**\n\n| Metric | Value |\n|---|---:|\n{body}\n"
    )


def render_svg(result: dict[str, Any]) -> str:
    counts = result["metrics"]["tenant_scheduled"]
    maximum = max(counts.values(), default=1)
    bars = []
    for index, (tenant, count) in enumerate(sorted(counts.items())):
        width = 440 * count / maximum
        y = 48 + index * 42
        bars.append(
            f'<text x="10" y="{y + 16}" font-size="13">{tenant}</text>'
            f'<rect x="120" y="{y}" width="{width:.1f}" height="22" fill="#4f46e5"/>'
            f'<text x="{128 + width:.1f}" y="{y + 16}" font-size="13">{count}</text>'
        )
    height = 75 + len(counts) * 42
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{height}" '
        'role="img" aria-label="Simulated requests scheduled by tenant">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="10" y="25" font-size="16" font-weight="bold">'
        'Simulated requests scheduled by tenant</text>' + "".join(bars) + "</svg>\n"
    )


def write_report(result: dict[str, Any], output_directory: Path) -> list[str]:
    failures = check_gates(result)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_directory / "report.md").write_text(render_markdown(result, failures))
    (output_directory / "tenant_fairness.svg").write_text(render_svg(result))
    return failures
