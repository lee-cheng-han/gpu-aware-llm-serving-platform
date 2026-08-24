from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from serving_platform.domain import ModelDefinition, RequestRecord, WorkerState


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    code: str
    reason: str


class AdmissionController(Protocol):
    def decide(
        self,
        request: RequestRecord,
        model: ModelDefinition | None,
        workers: Sequence[WorkerState],
    ) -> AdmissionDecision: ...
    def mark_running(self, request_id: str) -> None: ...
    def release(self, request_id: str) -> None: ...
    def snapshot(self, tenant_id: str) -> dict[str, int]: ...
