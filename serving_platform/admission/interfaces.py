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
