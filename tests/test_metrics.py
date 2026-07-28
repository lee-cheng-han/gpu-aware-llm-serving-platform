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


def test_cancelled_request_is_counted_once():
    metrics = Metrics()
    item = InferenceRequest("x", 1, 0, 1, "no_batching")
    item.queued_at = 1
    item.cancel()
    metrics.record(item)
    metrics.record(item)
    assert metrics.snapshot()["cancelled_requests"] == 1


def test_phase_metrics_and_recent_throughput():
    metrics = Metrics()
    item = InferenceRequest("x", 1, 0, 1, "no_batching")
    item.queued_at, item.started_at, item.completed_at = 1, 1.02, 1.1
    item.status, item.output_tokens = RequestStatus.COMPLETED, 5
    item.validation_tokenization_ms = 0.5
    item.worker_tokenization_ms = 1
    item.generation_ms = 7
    item.decoding_ms = 0.25
    metrics.record_model_invocation(batch_size=1, collection_ms=0)
    metrics.record(item)
    data = metrics.snapshot(queued=2, active=1, max_queue_depth=4)
    assert data["p50_queue_wait_ms"] == pytest.approx(20)
    assert data["avg_validation_tokenization_ms"] == 0.5
    assert data["avg_worker_tokenization_ms"] == 1
    assert data["avg_generation_ms"] == 7
    assert data["avg_decoding_ms"] == 0.25
    assert data["model_invocations"] == 1
    assert data["max_queue_depth"] == 4
    assert data["recent_requests_per_second"] > 0
    assert data["recent_tokens_per_second"] > 0
