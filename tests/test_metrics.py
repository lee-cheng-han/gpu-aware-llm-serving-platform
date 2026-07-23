import pytest

from app.metrics import Metrics, percentile
from scheduler.request import InferenceRequest, RequestStatus


def test_percentile_interpolates():
    assert percentile([], .5) == 0
    assert percentile([10], .95) == 10
    assert percentile([0, 10], .5) == 5


def test_metrics_counters_and_average():
    metrics = Metrics()
    item = InferenceRequest("x", 1, .7, 1, "dynamic_batch")
    item.queued_at, item.started_at, item.completed_at = 1, 1.01, 1.1
    item.status, item.batch_size, item.output_tokens = RequestStatus.COMPLETED, 4, 2
    metrics.received()
    metrics.record(item)
    data = metrics.snapshot()
    assert data["completed_requests"] == 1
    assert data["avg_batch_size"] == 4
    assert data["avg_queue_wait_ms"] == pytest.approx(10)
