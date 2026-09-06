from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations


class TopologyLinkType(StrEnum):
    CUDA_PEER = "cuda_peer"
    PCIE_ROOT = "pcie_root"
    HOST = "host"
    FABRIC = "fabric"
    NETWORK = "network"


DEFAULT_LINK_COST = {
    TopologyLinkType.CUDA_PEER: 1.0,
    TopologyLinkType.PCIE_ROOT: 4.0,
    TopologyLinkType.HOST: 12.0,
    TopologyLinkType.FABRIC: 25.0,
    TopologyLinkType.NETWORK: 100.0,
}


@dataclass(frozen=True)
class TopologyEndpoint:
    worker_id: str
    node_id: str | None
    device_index: int | None
    numa_node: int | None
    pcie_root_id: str | None
    fabric_id: str | None


@dataclass(frozen=True)
class TopologyLink:
    source_worker_id: str
    target_worker_id: str
    link_type: TopologyLinkType
    cost: float
    measured_bandwidth_gbps: float | None = None

    def __post_init__(self) -> None:
        if self.source_worker_id == self.target_worker_id or self.cost <= 0:
            raise ValueError("topology links require distinct endpoints and positive cost")
        if self.measured_bandwidth_gbps is not None and self.measured_bandwidth_gbps <= 0:
            raise ValueError("measured topology bandwidth must be positive")


class TopologyGraph:
    def __init__(
        self,
        endpoints: tuple[TopologyEndpoint, ...],
        links: tuple[TopologyLink, ...],
    ) -> None:
        self.endpoints = {endpoint.worker_id: endpoint for endpoint in endpoints}
        if len(self.endpoints) != len(endpoints):
            raise ValueError("topology endpoint identifiers must be unique")
        self._adjacency: dict[str, list[tuple[str, float]]] = {
            worker_id: [] for worker_id in self.endpoints
        }
        seen: set[frozenset[str]] = set()
        for link in links:
            if (
                link.source_worker_id not in self.endpoints
                or link.target_worker_id not in self.endpoints
            ):
                raise ValueError("topology link references an unknown endpoint")
            key = frozenset((link.source_worker_id, link.target_worker_id))
            if key in seen:
                raise ValueError("topology links must be unique per endpoint pair")
            seen.add(key)
            self._adjacency[link.source_worker_id].append(
                (link.target_worker_id, link.cost)
            )
            self._adjacency[link.target_worker_id].append(
                (link.source_worker_id, link.cost)
            )
        self.links = links

    def path_cost(self, source: str, target: str) -> float:
        if source not in self.endpoints or target not in self.endpoints:
            raise KeyError("unknown topology endpoint")
        if source == target:
            return 0.0
        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            cost, current = heapq.heappop(queue)
            if current == target:
                return cost
            if cost != distances[current]:
                continue
            for neighbor, edge_cost in self._adjacency[current]:
                candidate = cost + edge_cost
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        return float("inf")

    def group_cost(self, worker_ids: tuple[str, ...]) -> float:
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("topology group members must be unique")
        return sum(self.path_cost(first, second) for first, second in combinations(worker_ids, 2))
