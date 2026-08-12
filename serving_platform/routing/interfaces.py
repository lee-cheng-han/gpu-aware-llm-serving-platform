from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from serving_platform.domain import RequestRecord, WorkerState


@dataclass(frozen=True)
class RoutingDecision:
    selected_worker_id: str
    policy: str
    candidates: tuple[str, ...]
    rejected: dict[str, str]
    scores: dict[str, float]
    scoring_inputs: dict[str, dict[str, float | int | bool | None]]


class RoutingPolicy(Protocol):
    def select(
        self, request: RequestRecord, workers: Sequence[WorkerState]
    ) -> RoutingDecision: ...
