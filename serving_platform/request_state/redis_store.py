from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Protocol

from serving_platform.domain import RequestRecord, RequestState


class RedisClient(Protocol):
    def get(self, key: str) -> bytes | str | None: ...
    def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        xx: bool = False,
    ) -> object: ...
    def scan_iter(self, match: str) -> Iterable[bytes | str]: ...


class RedisRequestStateStore:
    """Redis-backed metadata store that redacts prompts and never stores output text."""

    def __init__(self, client: RedisClient, key_prefix: str = "llm:request:"):
        if not key_prefix:
            raise ValueError("Redis key prefix is required")
        self.client = client
        self.key_prefix = key_prefix

    def _key(self, request_id: str) -> str:
        return f"{self.key_prefix}{request_id}"

    @staticmethod
    def _encode(request: RequestRecord) -> str:
        return json.dumps(
            {
                "request_id": request.request_id,
                "tenant_id": request.tenant_id,
                "model_id": request.model_id,
                "prompt_tokens": request.prompt_tokens,
                "max_new_tokens": request.max_new_tokens,
                "priority": request.priority,
                "deadline": request.deadline,
                "stream": request.stream,
                "temperature": request.temperature,
                "created_at": request.created_at,
                "status": request.status.value,
                "assigned_worker_id": request.assigned_worker_id,
                "attempt_count": request.attempt_count,
                "retry_reasons": request.retry_reasons,
                "transition_timestamps": {
                    state.value: timestamp
                    for state, timestamp in request.transition_timestamps.items()
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _decode(value: bytes | str) -> RequestRecord:
        if isinstance(value, bytes):
            value = value.decode()
        payload = json.loads(value)
        return RequestRecord(
            request_id=payload["request_id"],
            tenant_id=payload["tenant_id"],
            model_id=payload["model_id"],
            prompt="[redacted]",
            prompt_tokens=payload["prompt_tokens"],
            max_new_tokens=payload["max_new_tokens"],
            priority=payload["priority"],
            deadline=payload["deadline"],
            stream=payload["stream"],
            temperature=payload["temperature"],
            created_at=payload["created_at"],
            status=RequestState(payload["status"]),
            assigned_worker_id=payload["assigned_worker_id"],
            attempt_count=payload["attempt_count"],
            retry_reasons=list(payload["retry_reasons"]),
            payload_available=False,
            transition_timestamps={
                RequestState(state): timestamp
                for state, timestamp in payload["transition_timestamps"].items()
            },
        )

    def create(self, request: RequestRecord) -> None:
        if not self.client.set(self._key(request.request_id), self._encode(request), nx=True):
            raise ValueError(f"request already exists: {request.request_id}")

    def get(self, request_id: str) -> RequestRecord | None:
        value = self.client.get(self._key(request_id))
        return None if value is None else self._decode(value)

    def save(self, request: RequestRecord) -> None:
        if not self.client.set(self._key(request.request_id), self._encode(request), xx=True):
            raise KeyError(f"request does not exist: {request.request_id}")

    def list(self) -> Sequence[RequestRecord]:
        records = [
            self._decode(value)
            for key in self.client.scan_iter(match=f"{self.key_prefix}*")
            if (value := self.client.get(key.decode() if isinstance(key, bytes) else key))
            is not None
        ]
        return tuple(sorted(records, key=lambda request: request.request_id))
