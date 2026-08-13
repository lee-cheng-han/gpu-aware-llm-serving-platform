from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class DeviceType(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    SIMULATED_GPU = "simulated_gpu"


class HealthStatus(StrEnum):
    REGISTERING = "registering"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    STOPPED = "stopped"


class RuntimeType(StrEnum):
    HUGGINGFACE = "huggingface"
    SIMULATED = "simulated"


class RequestState(StrEnum):
    RECEIVED = "received"
    VALIDATED = "validated"
    ADMITTED = "admitted"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    STREAMING = "streaming"
    COMPLETED = "completed"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_REQUEST_STATES = {
    RequestState.COMPLETED,
    RequestState.REJECTED,
    RequestState.TIMED_OUT,
    RequestState.CANCELLED,
    RequestState.FAILED,
}

ALLOWED_REQUEST_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    RequestState.RECEIVED: frozenset({
        RequestState.VALIDATED,
        RequestState.REJECTED,
        RequestState.CANCELLED,
    }),
    RequestState.VALIDATED: frozenset({
        RequestState.ADMITTED,
        RequestState.REJECTED,
        RequestState.TIMED_OUT,
        RequestState.CANCELLED,
    }),
    RequestState.ADMITTED: frozenset({
        RequestState.QUEUED,
        RequestState.ASSIGNED,
        RequestState.REJECTED,
        RequestState.TIMED_OUT,
        RequestState.CANCELLED,
    }),
    RequestState.QUEUED: frozenset({
        RequestState.ASSIGNED,
        RequestState.RUNNING,
        RequestState.REJECTED,
        RequestState.TIMED_OUT,
        RequestState.CANCELLED,
        RequestState.FAILED,
    }),
    RequestState.ASSIGNED: frozenset({
        RequestState.QUEUED,
        RequestState.RUNNING,
        RequestState.TIMED_OUT,
        RequestState.CANCELLED,
        RequestState.FAILED,
    }),
    RequestState.RUNNING: frozenset({
        RequestState.STREAMING,
        RequestState.COMPLETED,
        RequestState.TIMED_OUT,
        RequestState.CANCELLED,
        RequestState.FAILED,
    }),
    RequestState.STREAMING: frozenset({
        RequestState.COMPLETED,
        RequestState.TIMED_OUT,
        RequestState.CANCELLED,
        RequestState.FAILED,
    }),
    **{state: frozenset() for state in TERMINAL_REQUEST_STATES},
}


class InvalidStateTransition(ValueError):
    """Raised when a request lifecycle transition violates the state machine."""


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    revision: str
    runtime_type: RuntimeType
    estimated_memory_bytes: int
    supported_dtypes: tuple[str, ...]
    default_dtype: str
    max_context_tokens: int
    max_batch_tokens: int
    supports_streaming: bool
    supports_cancellation: bool
    load_timeout_seconds: float
    idle_eviction_seconds: float

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("model_id and revision are required")
        if self.default_dtype not in self.supported_dtypes:
            raise ValueError("default_dtype must be supported")
        if min(
            self.estimated_memory_bytes,
            self.max_context_tokens,
            self.max_batch_tokens,
        ) <= 0:
            raise ValueError("model memory and token limits must be positive")


@dataclass
class WorkerState:
    worker_id: str
    device_type: DeviceType
    device_name: str
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    resident_models: set[str] = field(default_factory=set)
    loading_models: set[str] = field(default_factory=set)
    queue_depth: int = 0
    active_batch_count: int = 0
    max_concurrency: int = 1
    recent_tokens_per_second: float = 0
    health_status: HealthStatus = HealthStatus.REGISTERING
    draining: bool = False
    last_heartbeat: float = field(default_factory=time.monotonic)
    allocated_memory_bytes: int | None = None
    reserved_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id is required")
        if (self.total_memory_bytes is None) != (self.available_memory_bytes is None):
            raise ValueError("worker memory values must both be known or both be unknown")
        if self.total_memory_bytes is None and (
            self.allocated_memory_bytes is not None
            or self.reserved_memory_bytes is not None
        ):
            raise ValueError("allocated and reserved memory require a known total")
        if self.total_memory_bytes is not None and self.available_memory_bytes is not None:
            if self.total_memory_bytes < 0 or not 0 <= self.available_memory_bytes <= self.total_memory_bytes:
                raise ValueError("worker memory accounting is invalid")
            for value in (self.allocated_memory_bytes, self.reserved_memory_bytes):
                if value is not None and not 0 <= value <= self.total_memory_bytes:
                    raise ValueError("worker CUDA memory accounting is invalid")
        if min(self.queue_depth, self.active_batch_count) < 0 or self.max_concurrency <= 0:
            raise ValueError("worker queue and concurrency values are invalid")


@dataclass
class RequestRecord:
    request_id: str
    tenant_id: str
    model_id: str
    prompt: str
    prompt_tokens: int
    max_new_tokens: int
    priority: int
    deadline: float
    stream: bool
    temperature: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    status: RequestState = RequestState.RECEIVED
    assigned_worker_id: str | None = None
    attempt_count: int = 0
    transition_timestamps: dict[RequestState, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.request_id, self.tenant_id, self.model_id)):
            raise ValueError("request, tenant, and model identifiers are required")
        if (
            not self.prompt.strip()
            or self.prompt_tokens < 0
            or self.max_new_tokens <= 0
            or self.temperature < 0
        ):
            raise ValueError("request token and prompt values are invalid")
        self.transition_timestamps.setdefault(self.status, self.created_at)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_REQUEST_STATES

    @property
    def estimated_tokens(self) -> int:
        return self.prompt_tokens + self.max_new_tokens

    def transition(self, target: RequestState, at: float | None = None) -> None:
        if target not in ALLOWED_REQUEST_TRANSITIONS[self.status]:
            raise InvalidStateTransition(f"cannot transition {self.status} to {target}")
        self.status = target
        self.transition_timestamps[target] = time.monotonic() if at is None else at


@dataclass(frozen=True)
class Assignment:
    request_id: str
    worker_id: str
    policy: str
    explanation: dict[str, object]
    assigned_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class TenantLimits:
    tenant_id: str
    max_concurrent_requests: int
    max_queued_requests: int
    token_quota: int
    scheduling_weight: int = 1

    def __post_init__(self) -> None:
        if not self.tenant_id or min(
            self.max_concurrent_requests,
            self.max_queued_requests,
            self.token_quota,
            self.scheduling_weight,
        ) <= 0:
            raise ValueError("tenant limits must be positive")
