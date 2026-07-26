import time
from threading import Lock


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


class Metrics:
    def __init__(self):
        self.started_at = time.monotonic()
        self.total_requests = self.completed_requests = 0
        self.failed_requests = self.rejected_requests = 0
        self.timeout_requests = self.queue_full_rejections = 0
        self.cancelled_requests = 0
        self.active_requests = self.queued_requests = 0
        self.latencies: list[float] = []
        self.ttfts: list[float] = []
        self.queue_waits: list[float] = []
        self.batch_sizes: list[int] = []
        self.output_tokens = 0
        self._lock = Lock()

    def received(self) -> None:
        with self._lock:
            self.total_requests += 1

    def rejected(self, queue_full: bool = False) -> None:
        with self._lock:
            self.rejected_requests += 1
            self.queue_full_rejections += int(queue_full)

    def record(self, request) -> None:
        with self._lock:
            if request.metrics_recorded:
                return
            request.metrics_recorded = True
            status = request.status.value
            self.completed_requests += status == "COMPLETED"
            self.timeout_requests += status == "TIMEOUT"
            self.failed_requests += status == "FAILED"
            self.cancelled_requests += status == "CANCELLED"
            if request.completed_at:
                self.latencies.append((request.completed_at - request.queued_at) * 1000)
            if request.started_at:
                self.queue_waits.append((request.started_at - request.queued_at) * 1000)
            if request.first_token_at:
                self.ttfts.append((request.first_token_at - request.queued_at) * 1000)
            self.batch_sizes.append(request.batch_size)
            self.output_tokens += request.output_tokens

    def snapshot(self, queued: int = 0, active: int = 0) -> dict:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "completed_requests": self.completed_requests,
                "failed_requests": self.failed_requests,
                "rejected_requests": self.rejected_requests,
                "timeout_requests": self.timeout_requests,
                "cancelled_requests": self.cancelled_requests,
                "queue_full_rejections": self.queue_full_rejections,
                "active_requests": active,
                "queued_requests": queued,
                "p50_latency_ms": percentile(self.latencies, .50),
                "p95_latency_ms": percentile(self.latencies, .95),
                "p50_ttft_ms": percentile(self.ttfts, .50),
                "p95_ttft_ms": percentile(self.ttfts, .95),
                "avg_queue_wait_ms": avg(self.queue_waits),
                "avg_batch_size": avg(self.batch_sizes),
                "requests_per_second": self.completed_requests / elapsed,
                "tokens_per_second": self.output_tokens / elapsed,
            }
