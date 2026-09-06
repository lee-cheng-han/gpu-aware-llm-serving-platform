import pytest

from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    RequestState,
    RuntimeType,
    WorkerState,
)
from serving_platform.registry import InMemoryModelRegistry
from serving_platform.routing import NoEligibleWorker
from serving_platform.topology import (
    TopologyAwareWorkerGroupPolicy,
    TopologyLinkType,
    discover_topology,
)


def model():
    return ModelDefinition(
        "model", "main", RuntimeType.SIMULATED, 100, ("float16",), "float16",
        100, 200, True, True, 5, 60,
    )


def request():
    item = RequestRecord(
        "request", "tenant", "model", "hello", 1, 2, 0, 100, False
    )
    item.transition(RequestState.VALIDATED)
    item.transition(RequestState.ADMITTED)
    return item


def worker(worker_id, node, index, root, *, queue=0, healthy=True):
    return WorkerState(
        worker_id,
        DeviceType.SIMULATED_GPU,
        "simulated",
        1000,
        800,
        resident_models={"model"},
        queue_depth=queue,
        health_status=(HealthStatus.HEALTHY if healthy else HealthStatus.SUSPECT),
        node_id=node,
        gpu_index=index,
        pcie_root_id=root,
    )


def test_discovery_prefers_proven_peer_access_without_claiming_nvlink():
    workers = [worker("a", "node-1", 0, "root-1"), worker("b", "node-1", 1, "root-1")]
    graph = discover_topology(workers, cuda_peer_access=lambda first, second: True)

    assert graph.links[0].link_type == TopologyLinkType.CUDA_PEER
    assert graph.links[0].measured_bandwidth_gbps is None
    assert graph.path_cost("a", "b") == 1


def test_topology_group_policy_balances_link_distance_and_worker_load():
    workers = [
        worker("a", "node-1", 0, "root-1"),
        worker("b", "node-1", 1, "root-1", queue=1),
        worker("c", "node-2", 0, "root-2"),
    ]
    graph = discover_topology(workers)
    policy = TopologyAwareWorkerGroupPolicy(
        InMemoryModelRegistry([model()]), graph, load_weight=1
    )

    decision = policy.select_group(request(), workers, group_size=2)

    assert decision.worker_ids == ("a", "b")
    assert decision.topology_cost == 4
    assert decision.load_cost == 1
    assert decision.candidates_considered == 3


def test_topology_policy_excludes_suspect_and_missing_inventory():
    healthy = worker("healthy", "node-1", 0, "root")
    suspect = worker("suspect", "node-1", 1, "root", healthy=False)
    missing = worker("missing", "node-1", 2, "root")
    graph = discover_topology([healthy, suspect])
    policy = TopologyAwareWorkerGroupPolicy(InMemoryModelRegistry([model()]), graph)

    with pytest.raises(NoEligibleWorker) as caught:
        policy.select_group(request(), [healthy, suspect, missing], group_size=2)

    assert caught.value.rejected == {
        "missing": "topology_missing",
        "suspect": "health:suspect",
    }
