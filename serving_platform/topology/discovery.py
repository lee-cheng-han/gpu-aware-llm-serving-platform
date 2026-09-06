from __future__ import annotations

from collections.abc import Callable, Sequence

from serving_platform.domain import WorkerState
from serving_platform.topology.models import (
    DEFAULT_LINK_COST,
    TopologyEndpoint,
    TopologyGraph,
    TopologyLink,
    TopologyLinkType,
)


def discover_topology(
    workers: Sequence[WorkerState],
    cuda_peer_access: Callable[[int, int], bool] | None = None,
) -> TopologyGraph:
    """Build a conservative graph without inventing NVLink or bandwidth measurements."""
    endpoints = tuple(
        TopologyEndpoint(
            worker.worker_id,
            worker.node_id,
            worker.gpu_index,
            worker.numa_node,
            worker.pcie_root_id,
            worker.fabric_id,
        )
        for worker in sorted(workers, key=lambda item: item.worker_id)
    )
    by_id = {worker.worker_id: worker for worker in workers}
    links: list[TopologyLink] = []
    for index, source in enumerate(endpoints):
        for target in endpoints[index + 1 :]:
            source_worker = by_id[source.worker_id]
            target_worker = by_id[target.worker_id]
            same_node = source.node_id is not None and source.node_id == target.node_id
            peer = (
                same_node
                and cuda_peer_access is not None
                and source.device_index is not None
                and target.device_index is not None
                and cuda_peer_access(source.device_index, target.device_index)
            )
            if peer:
                link_type = TopologyLinkType.CUDA_PEER
            elif (
                same_node
                and source.pcie_root_id is not None
                and source.pcie_root_id == target.pcie_root_id
            ):
                link_type = TopologyLinkType.PCIE_ROOT
            elif same_node:
                link_type = TopologyLinkType.HOST
            elif (
                source.fabric_id is not None
                and source.fabric_id == target.fabric_id
            ):
                link_type = TopologyLinkType.FABRIC
            else:
                link_type = TopologyLinkType.NETWORK
            links.append(
                TopologyLink(
                    source_worker.worker_id,
                    target_worker.worker_id,
                    link_type,
                    DEFAULT_LINK_COST[link_type],
                )
            )
    return TopologyGraph(endpoints, tuple(links))
