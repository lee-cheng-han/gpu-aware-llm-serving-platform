from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from inference.backend import BackendBundle
from inference.token_types import (
    FinishReason,
    GeneratedToken,
    GenerationComplete,
    SamplingParameters,
    SequencePhase,
    SequenceState,
)


class EngineCapacityError(RuntimeError):
    pass


class InvalidStopTokenError(ValueError):
    pass


class TokenGenerationEngine:
    """Single-device sequence engine; iteration batching is intentionally deferred."""

    def __init__(self, bundle: BackendBundle, max_active_sequences: int = 1):
        if max_active_sequences <= 0:
            raise ValueError("maximum active sequences must be positive")
        self.backend = bundle.backend
        self.executor = bundle.executor
        self.max_active_sequences = max_active_sequences
        self._active: dict[str, SequenceState] = {}
        self._closed = False

    def active_sequence_count(self) -> int:
        return len(self._active)

    def cancel(self, request_id: str) -> bool:
        sequence = self._active.get(request_id)
        if sequence is None or sequence.terminal:
            return False
        sequence.cancellation.set()
        return True

    def phase(self, request_id: str) -> SequencePhase | None:
        sequence = self._active.get(request_id)
        return sequence.phase if sequence else None

    def reserve(
        self,
        request_id: str,
        prompt: str,
        sampling: SamplingParameters,
        deadline: float,
    ) -> None:
        if self._closed:
            raise RuntimeError("generation engine is shut down")
        if request_id in self._active:
            raise ValueError(f"sequence already exists: {request_id}")
        if len(self._active) >= self.max_active_sequences:
            raise EngineCapacityError("token engine is at sequence capacity")
        self._active[request_id] = SequenceState(
            request_id,
            prompt,
            sampling,
            deadline,
        )

    async def validate_sampling(self, sampling: SamplingParameters) -> None:
        vocabulary_size = await self.executor.run_async(self.backend.vocabulary_size)
        if any(token_id >= vocabulary_size for token_id in sampling.stop_token_ids):
            raise InvalidStopTokenError("stop token identifier exceeds model vocabulary")

    async def count_prompt_tokens(self, prompt: str) -> int:
        tokenized = await self.executor.run_async(self.backend.tokenize, prompt)
        return len(tokenized.token_ids)

    @staticmethod
    def _finish_reason(sequence: SequenceState, token: GeneratedToken) -> FinishReason | None:
        if token.token_id in sequence.sampling.stop_token_ids:
            return FinishReason.STOP
        return None

    @staticmethod
    def _check_interruption(sequence: SequenceState) -> FinishReason | None:
        if sequence.cancellation.is_set():
            return FinishReason.CANCELLED
        if time.monotonic() >= sequence.deadline:
            return FinishReason.TIMEOUT
        return None

    async def _complete(
        self,
        sequence: SequenceState,
        reason: FinishReason,
    ) -> GenerationComplete:
        sequence.phase = {
            FinishReason.CANCELLED: SequencePhase.CANCELLED,
            FinishReason.TIMEOUT: SequencePhase.TIMED_OUT,
            FinishReason.FAILED: SequencePhase.FAILED,
        }.get(reason, SequencePhase.COMPLETED)
        return GenerationComplete(
            reason,
            len(sequence.prompt_token_ids),
            len(sequence.generated_token_ids),
            sequence.prefill_ms,
            sequence.decode_ms,
        )

    async def stream(
        self,
        request_id: str,
    ) -> AsyncIterator[GeneratedToken | GenerationComplete]:
        try:
            sequence = self._active[request_id]
        except KeyError as exc:
            raise KeyError(f"sequence is not reserved: {request_id}") from exc
        prompt = sequence.prompt
        sampling = sequence.sampling
        backend_sequence_created = False
        try:
            interrupted = self._check_interruption(sequence)
            if interrupted:
                yield await self._complete(sequence, interrupted)
                return
            sequence.phase = SequencePhase.PREFILL
            tokenized = await self.executor.run_async(self.backend.tokenize, prompt)
            sequence.prompt_token_ids = tokenized.token_ids
            started = time.perf_counter()
            # Release is idempotent, so mark ownership before entering the
            # non-preemptible call. If the awaiting task is cancelled while the
            # forward pass finishes, its newly created cache must still be freed.
            backend_sequence_created = True
            token = await self.executor.run_async(
                self.backend.prefill,
                request_id,
                tokenized,
                sampling,
            )
            sequence.prefill_ms = (time.perf_counter() - started) * 1000
            eos_ids = await self.executor.run_async(self.backend.eos_token_ids)
            interrupted = self._check_interruption(sequence)
            if interrupted:
                yield await self._complete(sequence, interrupted)
                return
            if token.token_id in eos_ids:
                yield await self._complete(sequence, FinishReason.EOS)
                return
            stopped = self._finish_reason(sequence, token)
            if stopped:
                yield await self._complete(sequence, stopped)
                return
            sequence.phase = SequencePhase.DECODE
            sequence.generated_token_ids.append(token.token_id)
            yield token

            while len(sequence.generated_token_ids) < sampling.max_new_tokens:
                interrupted = self._check_interruption(sequence)
                if interrupted:
                    yield await self._complete(sequence, interrupted)
                    return
                started = time.perf_counter()
                token = await self.executor.run_async(
                    self.backend.decode,
                    request_id,
                    sampling,
                    len(sequence.generated_token_ids),
                )
                sequence.decode_ms += (time.perf_counter() - started) * 1000
                interrupted = self._check_interruption(sequence)
                if interrupted:
                    yield await self._complete(sequence, interrupted)
                    return
                if token.token_id in eos_ids:
                    yield await self._complete(sequence, FinishReason.EOS)
                    return
                stopped = self._finish_reason(sequence, token)
                if stopped:
                    yield await self._complete(sequence, stopped)
                    return
                sequence.generated_token_ids.append(token.token_id)
                yield token
            yield await self._complete(sequence, FinishReason.LENGTH)
        except asyncio.CancelledError:
            sequence.cancellation.set()
            sequence.phase = SequencePhase.CANCELLED
            raise
        except BaseException:
            sequence.phase = SequencePhase.FAILED
            raise
        finally:
            if backend_sequence_created:
                await self.executor.run_async(self.backend.release_sequence, request_id)
            self._active.pop(request_id, None)

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sequence in self._active.values():
            sequence.cancellation.set()
        await self.executor.run_async(self.backend.shutdown)
        self.executor.shutdown()
