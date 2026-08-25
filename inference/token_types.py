from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event


class SequencePhase(StrEnum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class FinishReason(StrEnum):
    EOS = "eos"
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass(frozen=True)
class SamplingParameters:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0 or self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("sampling parameters are invalid")
        if len(self.stop_token_ids) > 32 or len(set(self.stop_token_ids)) != len(
            self.stop_token_ids
        ):
            raise ValueError("stop token identifiers must be unique and limited to 32")
        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError("stop token identifiers must be non-negative")


@dataclass(frozen=True)
class TokenizedPrompt:
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("tokenized prompt must not be empty")


@dataclass(frozen=True)
class GeneratedToken:
    token_id: int
    text: str
    index: int
    cache_tokens: int


@dataclass(frozen=True)
class GenerationComplete:
    finish_reason: FinishReason
    input_tokens: int
    output_tokens: int
    prefill_ms: float
    decode_ms: float


@dataclass
class SequenceState:
    request_id: str
    prompt: str
    sampling: SamplingParameters
    deadline: float
    phase: SequencePhase = SequencePhase.QUEUED
    prompt_token_ids: tuple[int, ...] = ()
    generated_token_ids: list[int] = field(default_factory=list)
    kv_block_ids: tuple[int, ...] = ()
    prefill_ms: float = 0
    decode_ms: float = 0
    created_at: float = field(default_factory=time.monotonic)
    cancellation: Event = field(default_factory=Event)

    @property
    def terminal(self) -> bool:
        return self.phase in {
            SequencePhase.COMPLETED,
            SequencePhase.CANCELLED,
            SequencePhase.TIMED_OUT,
            SequencePhase.FAILED,
        }
