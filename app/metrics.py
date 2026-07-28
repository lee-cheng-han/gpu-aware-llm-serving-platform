import time
from collections import Counter, deque
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
        self.validation_tokenization_times: list[float] = []
        self.worker_tokenization_times: list[float] = []
        self.generation_times: list[float] = []
        self.decoding_times: list[float] = []
        self.batch_collection_times: list[float] = []
        self.model_invocations = 0
        self.batches = 0
        self.output_tokens = 0
        self._recent_completions: deque[tuple[float, int]] = deque()
        self._lock = Lock()

    def received(self) -> None:
        with self._lock:
            self.total_requests += 1

    def rejected(self, queue_full: bool = False) -> None:
        with self._lock:
            self.rejected_requests += 1
            self.queue_full_rejections += int(queue_full)

    def record_model_invocation(self, batch_size: int, collection_ms: float) -> None:
        with self._lock:
            self.model_invocations += 1
            self.batches += 1
            self.batch_collection_times.append(collection_ms)

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
            if request.started_at:
                self.batch_sizes.append(request.batch_size)
            self.validation_tokenization_times.append(request.validation_tokenization_ms)
            if request.started_at:
                self.worker_tokenization_times.append(request.worker_tokenization_ms)
                self.generation_times.append(request.generation_ms)
                self.decoding_times.append(request.decoding_ms)
            self.output_tokens += request.output_tokens
            if status == "COMPLETED":
                self._recent_completions.append((time.monotonic(), request.output_tokens))

    def snapshot(
        self,
        queued: int = 0,
        active: int = 0,
        max_queue_depth: int = 0,
    ) -> dict:
        now = time.monotonic()
        elapsed = max(now - self.started_at, 1e-9)
        recent_window_seconds = min(elapsed, 60.0)
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        with self._lock:
            cutoff = now - 60
            while self._recent_completions and self._recent_completions[0][0] < cutoff:
                self._recent_completions.popleft()
            recent_requests = len(self._recent_completions)
            recent_tokens = sum(tokens for _, tokens in self._recent_completions)
            batch_histogram = {
                str(size): count for size, count in sorted(Counter(self.batch_sizes).items())
            }
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
                "max_queue_depth": max_queue_depth,
                "p50_latency_ms": percentile(self.latencies, .50),
                "p95_latency_ms": percentile(self.latencies, .95),
                "p50_queue_wait_ms": percentile(self.queue_waits, .50),
                "p95_queue_wait_ms": percentile(self.queue_waits, .95),
                "p50_ttft_ms": percentile(self.ttfts, .50),
                "p95_ttft_ms": percentile(self.ttfts, .95),
                "avg_queue_wait_ms": avg(self.queue_waits),
                "avg_batch_size": avg(self.batch_sizes),
                "batch_size_histogram": batch_histogram,
                "model_invocations": self.model_invocations,
                "batches": self.batches,
                "avg_validation_tokenization_ms": avg(self.validation_tokenization_times),
                "avg_worker_tokenization_ms": avg(self.worker_tokenization_times),
                "avg_generation_ms": avg(self.generation_times),
                "avg_decoding_ms": avg(self.decoding_times),
                "avg_batch_collection_ms": avg(self.batch_collection_times),
                "requests_per_second": self.completed_requests / elapsed,
                "tokens_per_second": self.output_tokens / elapsed,
                "recent_window_seconds": recent_window_seconds,
                "recent_requests_per_second": recent_requests / recent_window_seconds,
                "recent_tokens_per_second": recent_tokens / recent_window_seconds,
            }
