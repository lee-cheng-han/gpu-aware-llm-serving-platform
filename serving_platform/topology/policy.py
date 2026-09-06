from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from serving_platform.domain import RequestRecord, WorkerState
from serving_platform.registry.interfaces import ModelRegistry
from serving_platform.routing.policies import NoEligibleWorker, PlacementPolicy
from serving_platform.topology.models import TopologyGraph


@dataclass(frozen=True)
class WorkerGroupDecision:
    worker_ids: tuple[str, ...]
    policy: str
    topology_cost: float
    load_cost: float
    candidates_considered: int
    rejected_workers: dict[str, str]


class TopologyAwareWorkerGroupPolicy(PlacementPolicy):
    name = "topology_aware_worker_group"

    def __init__(
        self,
        models: ModelRegistry,
        topology: TopologyGraph,
        load_weight: float = 1.0,
        memory_safety_reserve_bytes: int = 0,
    ) -> None:
        super().__init__(models, memory_safety_reserve_bytes)
        if load_weight < 0:
            raise ValueError("topology load weight must be non-negative")
        self.topology = topology
        self.load_weight = load_weight

    def select_group(
        self,
        request: RequestRecord,
        workers: Sequence[WorkerState],
        group_size: int,
    ) -> WorkerGroupDecision:
        if group_size <= 0:
            raise ValueError("worker group size must be positive")
        _model, eligible, rejected = self._filter(request, workers)
        topology_eligible = []
        for worker in eligible:
            if worker.worker_id not in self.topology.endpoints:
                rejected[worker.worker_id] = "topology_missing"
            else:
                topology_eligible.append(worker)
        eligible = tuple(topology_eligible)
        if len(eligible) < group_size:
            raise NoEligibleWorker(rejected)
        ranked: list[tuple[float, tuple[str, ...], float, float]] = []
        for group in combinations(eligible, group_size):
            worker_ids = tuple(sorted(worker.worker_id for worker in group))
            topology_cost = self.topology.group_cost(worker_ids)
            if topology_cost == float("inf"):
                continue
            load_cost = sum(
                worker.queue_depth
                + worker.active_batch_count / worker.max_concurrency
                for worker in group
            )
            ranked.append(
                (
                    topology_cost + self.load_weight * load_cost,
                    worker_ids,
                    topology_cost,
                    load_cost,
                )
            )
        if not ranked:
            raise NoEligibleWorker(rejected)
        _, worker_ids, topology_cost, load_cost = min(ranked)
        return WorkerGroupDecision(
            worker_ids,
            self.name,
            topology_cost,
            load_cost,
            len(ranked),
            rejected,
        )
