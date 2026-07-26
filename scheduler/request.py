import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid


class RequestStatus(str, Enum):
    RECEIVED = "RECEIVED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = {
    RequestStatus.COMPLETED,
    RequestStatus.FAILED,
    RequestStatus.REJECTED,
    RequestStatus.TIMEOUT,
    RequestStatus.CANCELLED,
}


@dataclass
class InferenceRequest:
    prompt: str
    max_new_tokens: int
    temperature: float
    estimated_tokens: int
    scheduler_policy: str
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex}")
    created_at: float = field(default_factory=time.monotonic)
    queued_at: float = 0
    started_at: float = 0
    first_token_at: float = 0
    completed_at: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    status: RequestStatus = RequestStatus.RECEIVED
    result_text: str = ""
    error_message: str = ""
    batch_size: int = 1
    future: asyncio.Future | None = None
    metrics_recorded: bool = False

    def timed_out(self, timeout: float) -> bool:
        return bool(self.queued_at and time.monotonic() - self.queued_at >= timeout)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def finish(self, status: RequestStatus, error_message: str = "") -> None:
        self.status = status
        self.error_message = error_message
        self.completed_at = self.completed_at or time.monotonic()
        if self.future and not self.future.done():
            self.future.set_result(self)

    def cancel(self, reason: str = "client disconnected") -> None:
        if not self.terminal:
            self.finish(RequestStatus.CANCELLED, reason)
