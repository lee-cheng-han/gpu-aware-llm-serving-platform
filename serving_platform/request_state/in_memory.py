from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from threading import RLock

from serving_platform.domain import RequestRecord


class InMemoryRequestStateStore:
    def __init__(self) -> None:
        self._requests: dict[str, RequestRecord] = {}
        self._lock = RLock()

    def create(self, request: RequestRecord) -> None:
        with self._lock:
            if request.request_id in self._requests:
                raise ValueError(f"request already exists: {request.request_id}")
            self._requests[request.request_id] = deepcopy(request)

    def get(self, request_id: str) -> RequestRecord | None:
        with self._lock:
            request = self._requests.get(request_id)
            return deepcopy(request) if request else None

    def save(self, request: RequestRecord) -> None:
        with self._lock:
            if request.request_id not in self._requests:
                raise KeyError(f"request does not exist: {request.request_id}")
            self._requests[request.request_id] = deepcopy(request)

    def list(self) -> Sequence[RequestRecord]:
        with self._lock:
            return tuple(deepcopy(self._requests[key]) for key in sorted(self._requests))
