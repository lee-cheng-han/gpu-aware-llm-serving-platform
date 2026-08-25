from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from inference.token_types import SequenceState


class TokenSchedulingPolicyName(StrEnum):
    PREFILL_FIRST = "prefill_first"
    DECODE_FIRST = "decode_first"
    BALANCED = "balanced"


@dataclass(frozen=True)
class TokenResourceSnapshot:
    available_slots: int
    free_kv_blocks: int
    max_batch_size: int


@dataclass(frozen=True)
class TokenSchedulingDecision:
    prefill_ids: tuple[str, ...] = ()
    decode_ids: tuple[str, ...] = ()
    reason: str = ""


class TokenSchedulingPolicy(Protocol):
    def select(
        self,
        waiting: Sequence[SequenceState],
        running: Sequence[SequenceState],
        resources: TokenResourceSnapshot,
    ) -> TokenSchedulingDecision: ...


class PrefillDecodePolicy:
    """Selects one iteration workload without mixing compute-heavy prefill and decode."""

    def __init__(self, name: TokenSchedulingPolicyName | str = "decode_first") -> None:
        self.name = TokenSchedulingPolicyName(name)
        self._balanced_decode_turn = False

    def select(
        self,
        waiting: Sequence[SequenceState],
        running: Sequence[SequenceState],
        resources: TokenResourceSnapshot,
    ) -> TokenSchedulingDecision:
        limit = resources.max_batch_size
        waiting_ids = tuple(item.request_id for item in waiting[:limit])
        running_ids = tuple(item.request_id for item in running[:limit])
        if self.name == TokenSchedulingPolicyName.PREFILL_FIRST:
            return (
                TokenSchedulingDecision(prefill_ids=waiting_ids, reason="prefill_first")
                if waiting_ids
                else TokenSchedulingDecision(decode_ids=running_ids, reason="decode_fallback")
            )
        if self.name == TokenSchedulingPolicyName.DECODE_FIRST:
            return (
                TokenSchedulingDecision(decode_ids=running_ids, reason="decode_first")
                if running_ids
                else TokenSchedulingDecision(prefill_ids=waiting_ids, reason="prefill_fallback")
            )
        self._balanced_decode_turn = not self._balanced_decode_turn
        if self._balanced_decode_turn and running_ids:
            return TokenSchedulingDecision(decode_ids=running_ids, reason="balanced_decode")
        if waiting_ids:
            return TokenSchedulingDecision(prefill_ids=waiting_ids, reason="balanced_prefill")
        return TokenSchedulingDecision(decode_ids=running_ids, reason="balanced_decode_fallback")
