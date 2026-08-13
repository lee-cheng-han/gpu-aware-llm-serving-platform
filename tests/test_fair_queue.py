from serving_platform.domain import RequestRecord, RequestState, TenantLimits
from serving_platform.scheduling import WeightedFairRequestQueue


def request(request_id: str, tenant: str, priority: int = 0, tokens: int = 1):
    item = RequestRecord(
        request_id, tenant, "model", "hello", tokens - 1, 1, priority, 100, False
    )
    item.transition(RequestState.VALIDATED, 0)
    item.transition(RequestState.ADMITTED, 0)
    return item


def test_high_volume_tenant_cannot_starve_another_tenant():
    queue = WeightedFairRequestQueue(
        [TenantLimits("noisy", 10, 100, 1000), TenantLimits("quiet", 10, 100, 1000)],
        base_quantum_tokens=1,
        clock=lambda: 0,
    )
    for number in range(10):
        queue.enqueue(request(f"noisy-{number}", "noisy"))
    queue.enqueue(request("quiet", "quiet"))
    assert [queue.pop().tenant_id for _ in range(2)] == ["noisy", "quiet"]


def test_weight_and_priority_affect_order_without_permanent_starvation():
    now = [0.0]
    queue = WeightedFairRequestQueue(
        [TenantLimits("tenant", 10, 100, 1000, scheduling_weight=2)],
        base_quantum_tokens=1,
        priority_aging_seconds=1,
        clock=lambda: now[0],
    )
    low = request("low", "tenant", priority=0)
    queue.enqueue(low)
    now[0] = 5
    high = request("high", "tenant", priority=3)
    queue.enqueue(high)
    # The older low-priority request has aged above the newer high-priority request.
    assert queue.pop().request_id == "low"
    assert queue.pop().request_id == "high"


def test_fair_queue_cancellation_is_terminal():
    queue = WeightedFairRequestQueue([TenantLimits("tenant", 1, 2, 100)])
    item = request("request", "tenant")
    queue.enqueue(item)
    assert queue.cancel(item.request_id)
    assert item.status == RequestState.CANCELLED
    assert queue.pop() is None
