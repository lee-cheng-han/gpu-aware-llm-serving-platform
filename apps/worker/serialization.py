from __future__ import annotations

import math
from typing import Any

from runtime.base import RuntimeCapacity, RuntimeResult
from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    RequestState,
    RuntimeType,
    WorkerState,
)


def model_to_dict(model: ModelDefinition) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "revision": model.revision,
        "runtime_type": model.runtime_type.value,
        "estimated_memory_bytes": model.estimated_memory_bytes,
        "supported_dtypes": list(model.supported_dtypes),
        "default_dtype": model.default_dtype,
        "max_context_tokens": model.max_context_tokens,
        "max_batch_tokens": model.max_batch_tokens,
        "supports_streaming": model.supports_streaming,
        "supports_cancellation": model.supports_cancellation,
        "load_timeout_seconds": model.load_timeout_seconds,
        "idle_eviction_seconds": model.idle_eviction_seconds,
    }


def model_from_dict(payload: dict[str, Any]) -> ModelDefinition:
    return ModelDefinition(
        model_id=str(payload["model_id"]),
        revision=str(payload["revision"]),
        runtime_type=RuntimeType(payload["runtime_type"]),
        estimated_memory_bytes=int(payload["estimated_memory_bytes"]),
        supported_dtypes=tuple(payload["supported_dtypes"]),
        default_dtype=str(payload["default_dtype"]),
        max_context_tokens=int(payload["max_context_tokens"]),
        max_batch_tokens=int(payload["max_batch_tokens"]),
        supports_streaming=bool(payload["supports_streaming"]),
        supports_cancellation=bool(payload["supports_cancellation"]),
        load_timeout_seconds=float(payload["load_timeout_seconds"]),
        idle_eviction_seconds=float(payload["idle_eviction_seconds"]),
    )


def worker_state_to_dict(worker: WorkerState) -> dict[str, Any]:
    return {
        "worker_id": worker.worker_id,
        "device_type": worker.device_type.value,
        "device_name": worker.device_name,
        "total_memory_bytes": worker.total_memory_bytes,
        "available_memory_bytes": worker.available_memory_bytes,
        "resident_models": sorted(worker.resident_models),
        "loading_models": sorted(worker.loading_models),
        "queue_depth": worker.queue_depth,
        "active_batch_count": worker.active_batch_count,
        "max_concurrency": worker.max_concurrency,
        "recent_tokens_per_second": worker.recent_tokens_per_second,
        "health_status": worker.health_status.value,
        "draining": worker.draining,
        "last_heartbeat": worker.last_heartbeat,
        "allocated_memory_bytes": worker.allocated_memory_bytes,
        "reserved_memory_bytes": worker.reserved_memory_bytes,
    }


def worker_state_from_dict(payload: dict[str, Any]) -> WorkerState:
    return WorkerState(
        worker_id=str(payload["worker_id"]),
        device_type=DeviceType(payload["device_type"]),
        device_name=str(payload["device_name"]),
        total_memory_bytes=payload["total_memory_bytes"],
        available_memory_bytes=payload["available_memory_bytes"],
        resident_models=set(payload["resident_models"]),
        loading_models=set(payload["loading_models"]),
        queue_depth=int(payload["queue_depth"]),
        active_batch_count=int(payload["active_batch_count"]),
        max_concurrency=int(payload["max_concurrency"]),
        recent_tokens_per_second=float(payload["recent_tokens_per_second"]),
        health_status=HealthStatus(payload["health_status"]),
        draining=bool(payload["draining"]),
        last_heartbeat=float(payload["last_heartbeat"]),
        allocated_memory_bytes=payload["allocated_memory_bytes"],
        reserved_memory_bytes=payload["reserved_memory_bytes"],
    )


def request_to_dict(request: RequestRecord) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "tenant_id": request.tenant_id,
        "model_id": request.model_id,
        "prompt": request.prompt,
        "prompt_tokens": request.prompt_tokens,
        "max_new_tokens": request.max_new_tokens,
        "priority": request.priority,
        "deadline": request.deadline if math.isfinite(request.deadline) else None,
        "stream": request.stream,
        "temperature": request.temperature,
        "created_at": request.created_at,
        "status": request.status.value,
        "assigned_worker_id": request.assigned_worker_id,
        "attempt_count": request.attempt_count,
        "retry_reasons": list(request.retry_reasons),
        "payload_available": request.payload_available,
        "transition_timestamps": {
            state.value: timestamp
            for state, timestamp in request.transition_timestamps.items()
        },
    }


def request_from_dict(payload: dict[str, Any]) -> RequestRecord:
    return RequestRecord(
        request_id=str(payload["request_id"]),
        tenant_id=str(payload["tenant_id"]),
        model_id=str(payload["model_id"]),
        prompt=str(payload["prompt"]),
        prompt_tokens=int(payload["prompt_tokens"]),
        max_new_tokens=int(payload["max_new_tokens"]),
        priority=int(payload["priority"]),
        deadline=(
            float(payload["deadline"])
            if payload["deadline"] is not None
            else float("inf")
        ),
        stream=bool(payload["stream"]),
        temperature=float(payload["temperature"]),
        created_at=float(payload["created_at"]),
        status=RequestState(payload["status"]),
        assigned_worker_id=payload["assigned_worker_id"],
        attempt_count=int(payload["attempt_count"]),
        retry_reasons=list(payload["retry_reasons"]),
        payload_available=bool(payload["payload_available"]),
        transition_timestamps={
            RequestState(state): float(timestamp)
            for state, timestamp in payload["transition_timestamps"].items()
        },
    )


def apply_request_dict(request: RequestRecord, payload: dict[str, Any]) -> None:
    restored = request_from_dict(payload)
    request.status = restored.status
    request.assigned_worker_id = restored.assigned_worker_id
    request.attempt_count = restored.attempt_count
    request.retry_reasons = restored.retry_reasons
    request.payload_available = restored.payload_available
    request.transition_timestamps = restored.transition_timestamps


def capacity_to_dict(capacity: RuntimeCapacity) -> dict[str, Any]:
    return {
        "device_type": capacity.device_type.value,
        "device_name": capacity.device_name,
        "total_memory_bytes": capacity.total_memory_bytes,
        "available_memory_bytes": capacity.available_memory_bytes,
        "allocated_memory_bytes": capacity.allocated_memory_bytes,
        "reserved_memory_bytes": capacity.reserved_memory_bytes,
    }


def capacity_from_dict(payload: dict[str, Any]) -> RuntimeCapacity:
    return RuntimeCapacity(
        DeviceType(payload["device_type"]),
        str(payload["device_name"]),
        payload["total_memory_bytes"],
        payload["available_memory_bytes"],
        payload["allocated_memory_bytes"],
        payload["reserved_memory_bytes"],
    )


def result_to_dict(result: RuntimeResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "tokenization_ms": result.tokenization_ms,
        "generation_ms": result.generation_ms,
        "decoding_ms": result.decoding_ms,
    }


def result_from_dict(payload: dict[str, Any]) -> RuntimeResult:
    return RuntimeResult(
        text=str(payload["text"]),
        input_tokens=int(payload["input_tokens"]),
        output_tokens=int(payload["output_tokens"]),
        tokenization_ms=float(payload["tokenization_ms"]),
        generation_ms=float(payload["generation_ms"]),
        decoding_ms=float(payload["decoding_ms"]),
    )
