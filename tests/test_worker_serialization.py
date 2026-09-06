from apps.worker.serialization import (
    capacity_from_dict,
    capacity_to_dict,
    worker_state_from_dict,
    worker_state_to_dict,
)
from runtime.base import RuntimeCapacity
from serving_platform.domain import DeviceType, HealthStatus, WorkerState


def test_rich_worker_inventory_round_trips_over_transport():
    worker = WorkerState(
        "gpu-0",
        DeviceType.CUDA,
        "Example GPU",
        1000,
        750,
        health_status=HealthStatus.SUSPECT,
        allocated_memory_bytes=100,
        reserved_memory_bytes=200,
        gpu_index=0,
        compute_capability="9.0",
        driver_version="example-driver",
        runtime_version="12.8",
        heartbeat_sequence=7,
        consecutive_failures=2,
        last_error_code="heartbeat_delayed",
        node_id="node-a",
        numa_node=1,
        pcie_root_id="0000:40",
        fabric_id="rack-a",
    )
    assert worker_state_from_dict(worker_state_to_dict(worker)) == worker


def test_runtime_capacity_inventory_round_trips_over_transport():
    capacity = RuntimeCapacity(
        DeviceType.CUDA,
        "Example GPU",
        1000,
        750,
        100,
        200,
        0,
        "9.0",
        "example-driver",
        "12.8",
        1,
        "0000:40",
        "rack-a",
    )
    assert capacity_from_dict(capacity_to_dict(capacity)) == capacity
