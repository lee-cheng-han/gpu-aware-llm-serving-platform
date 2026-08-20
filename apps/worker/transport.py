from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from apps.worker.serialization import (
    apply_request_dict,
    capacity_from_dict,
    model_to_dict,
    request_to_dict,
    result_from_dict,
    worker_state_from_dict,
)
from apps.worker.service import WorkerExecutionResult
from runtime.base import RuntimeCapacity
from serving_platform.domain import HealthStatus, ModelDefinition, RequestRecord, WorkerState
from serving_platform.registry.interfaces import WorkerRegistry


class WorkerTransportError(RuntimeError):
    pass


class HttpWorkerClient:
    """Synchronous control-plane handle for a worker in another local process."""

    def __init__(
        self,
        worker_id: str,
        base_url: str,
        auth_token: str,
        registry: WorkerRegistry,
        timeout_seconds: float = 5,
    ):
        if not worker_id or not base_url or not auth_token or timeout_seconds <= 0:
            raise ValueError("remote worker transport settings are invalid")
        self.worker_id = worker_id
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self._requests: dict[str, RequestRecord] = {}

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "X-Worker-Token": self.auth_token,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise WorkerTransportError(
                f"worker {self.worker_id} transport request failed: {method} {path}"
            ) from exc

    def register(self) -> WorkerState:
        snapshot = worker_state_from_dict(self._request("POST", "/internal/register"))
        if snapshot.worker_id != self.worker_id:
            raise WorkerTransportError("worker registration identity did not match")
        self.registry.register(snapshot)
        return snapshot

    def heartbeat(self) -> WorkerState:
        snapshot = worker_state_from_dict(self._request("POST", "/internal/heartbeat"))
        if snapshot.worker_id != self.worker_id:
            raise WorkerTransportError("worker heartbeat identity did not match")
        self.registry.heartbeat(snapshot)
        return snapshot

    def load_model(self, model: ModelDefinition) -> None:
        self._request("POST", "/internal/models/load", model_to_dict(model))

    def unload_model(self, model_id: str) -> None:
        encoded = urllib.parse.quote(model_id, safe="")
        self._request("DELETE", f"/internal/models/{encoded}")
        self.heartbeat()

    def warmup_model(self, model_id: str) -> None:
        encoded = urllib.parse.quote(model_id, safe="")
        self._request("POST", f"/internal/models/{encoded}/warmup")

    def enqueue_request(self, request: RequestRecord) -> None:
        payload = self._request("POST", "/internal/requests", request_to_dict(request))
        apply_request_dict(request, payload)
        self._requests[request.request_id] = request

    def cancel_request(self, request_id: str) -> bool:
        encoded = urllib.parse.quote(request_id, safe="")
        payload = self._request("DELETE", f"/internal/requests/{encoded}")
        request_payload = payload.get("request")
        if request_payload is not None and request_id in self._requests:
            apply_request_dict(self._requests[request_id], request_payload)
        return bool(payload["cancelled"])

    def execute_batch(self) -> tuple[WorkerExecutionResult, ...]:
        payload = self._request("POST", "/internal/batches/execute")
        executions: list[WorkerExecutionResult] = []
        for item in payload["executions"]:
            request_id = str(item["request_id"])
            request = self._requests.get(request_id)
            if request is None:
                raise WorkerTransportError("worker returned an unknown request")
            apply_request_dict(request, item["request"])
            executions.append(
                WorkerExecutionResult(request_id, result_from_dict(item["result"]))
            )
        return tuple(executions)

    def drain(self) -> None:
        self._request("POST", "/internal/drain")
        self.heartbeat()

    def shutdown(self) -> None:
        self._request("POST", "/internal/shutdown")
        self.registry.unregister(self.worker_id)

    def health(self) -> HealthStatus:
        return HealthStatus(self._request("GET", "/internal/health")["status"])

    def capacity(self) -> RuntimeCapacity:
        return capacity_from_dict(self._request("GET", "/internal/capacity"))
