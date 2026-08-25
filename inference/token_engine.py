from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from inference.backend import BackendBundle, BatchTokenResult
from inference.kv_cache import KVBlockAllocator, KVCacheCapacityError, KVCacheSnapshot
from inference.token_scheduling import (
    PrefillDecodePolicy,
    TokenResourceSnapshot,
    TokenSchedulingDecision,
)
from inference.token_types import (
    FinishReason,
    GeneratedToken,
    GenerationComplete,
    SamplingParameters,
    SequencePhase,
    SequenceState,
    TokenizedPrompt,
)


class EngineCapacityError(RuntimeError):
    pass


class InvalidStopTokenError(ValueError):
    pass


@dataclass(frozen=True)
class TokenSchedulerSnapshot:
    waiting_sequences: int
    running_sequences: int
    scheduler_iterations: int
    prefill_batches: int
    decode_batches: int
    max_observed_batch_size: int
    policy: str
    kv_cache: KVCacheSnapshot


@dataclass(frozen=True)
class _SequenceFailure:
    error: BaseException


@dataclass
class _ScheduledSequence:
    state: SequenceState
    events: asyncio.Queue[GeneratedToken | GenerationComplete | _SequenceFailure] = field(
        default_factory=asyncio.Queue
    )
    done: asyncio.Event = field(default_factory=asyncio.Event)
    tokenized: TokenizedPrompt | None = None
    backend_created: bool = False
    cache_tokens: int = 0
    prefill_offset: int = 0


class TokenGenerationEngine:
    """Iteration scheduler with waiting/running sets and dynamically joining sequences."""

    def __init__(
        self,
        bundle: BackendBundle,
        max_active_sequences: int = 1,
        *,
        max_batch_size: int | None = None,
        scheduling_policy: str = "decode_first",
        kv_allocator: KVBlockAllocator | None = None,
        prefill_chunk_size_tokens: int = 0,
    ) -> None:
        if max_active_sequences <= 0:
            raise ValueError("maximum active sequences must be positive")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("token batch size must be positive")
        if prefill_chunk_size_tokens < 0:
            raise ValueError("prefill chunk size must be non-negative")
        self.backend = bundle.backend
        self.executor = bundle.executor
        self.max_active_sequences = max_active_sequences
        self.max_batch_size = max_batch_size or max_active_sequences
        self.policy = PrefillDecodePolicy(scheduling_policy)
        self.prefill_chunk_size_tokens = prefill_chunk_size_tokens
        self.kv_allocator = kv_allocator or KVBlockAllocator(
            total_blocks=max(1024, max_active_sequences * 16),
            block_size_tokens=16,
        )
        self._active: dict[str, _ScheduledSequence] = {}
        self._finished: dict[str, _ScheduledSequence] = {}
        self._scheduler_task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._closed = False
        self._eos_ids: frozenset[int] | None = None
        self._iterations = 0
        self._prefill_batches = 0
        self._decode_batches = 0
        self._max_observed_batch_size = 0

    def active_sequence_count(self) -> int:
        return len(self._active)

    def cancel(self, request_id: str) -> bool:
        runtime = self._active.get(request_id)
        if runtime is None or runtime.state.terminal:
            return False
        runtime.state.cancellation.set()
        self._wake.set()
        return True

    def phase(self, request_id: str) -> SequencePhase | None:
        runtime = self._active.get(request_id)
        return runtime.state.phase if runtime else None

    def reserve(
        self,
        request_id: str,
        prompt: str,
        sampling: SamplingParameters,
        deadline: float,
        prompt_token_count: int = 1,
    ) -> None:
        if self._closed:
            raise RuntimeError("generation engine is shut down")
        if request_id in self._active or request_id in self._finished:
            raise ValueError(f"sequence already exists: {request_id}")
        if len(self._active) >= self.max_active_sequences:
            raise EngineCapacityError("token engine is at sequence capacity")
        try:
            blocks = self.kv_allocator.allocate(
                request_id, prompt_token_count + sampling.max_new_tokens
            )
        except KVCacheCapacityError as exc:
            raise EngineCapacityError(str(exc)) from exc
        state = SequenceState(request_id, prompt, sampling, deadline)
        state.kv_block_ids = blocks
        self._active[request_id] = _ScheduledSequence(state)
        self._wake.set()

    async def validate_sampling(self, sampling: SamplingParameters) -> None:
        vocabulary_size = await self.executor.run_async(self.backend.vocabulary_size)
        if any(token_id >= vocabulary_size for token_id in sampling.stop_token_ids):
            raise InvalidStopTokenError("stop token identifier exceeds model vocabulary")

    async def count_prompt_tokens(self, prompt: str) -> int:
        tokenized = await self.executor.run_async(self.backend.tokenize, prompt)
        return len(tokenized.token_ids)

    def scheduler_snapshot(self) -> TokenSchedulerSnapshot:
        waiting = sum(
            runtime.state.phase == SequencePhase.QUEUED
            for runtime in self._active.values()
        )
        running = sum(
            runtime.state.phase in {SequencePhase.PREFILL, SequencePhase.DECODE}
            for runtime in self._active.values()
        )
        return TokenSchedulerSnapshot(
            waiting,
            running,
            self._iterations,
            self._prefill_batches,
            self._decode_batches,
            self._max_observed_batch_size,
            self.policy.name.value,
            self.kv_allocator.snapshot(),
        )

    @staticmethod
    def _check_interruption(sequence: SequenceState) -> FinishReason | None:
        if sequence.cancellation.is_set():
            return FinishReason.CANCELLED
        if time.monotonic() >= sequence.deadline:
            return FinishReason.TIMEOUT
        return None

    @staticmethod
    def _phase_for(reason: FinishReason) -> SequencePhase:
        return {
            FinishReason.CANCELLED: SequencePhase.CANCELLED,
            FinishReason.TIMEOUT: SequencePhase.TIMED_OUT,
            FinishReason.FAILED: SequencePhase.FAILED,
        }.get(reason, SequencePhase.COMPLETED)

    def _ensure_scheduler(self) -> None:
        if self._scheduler_task is None:
            self._scheduler_task = asyncio.create_task(
                self._scheduler_loop(), name="continuous-token-scheduler"
            )

    def _sets(self) -> tuple[list[_ScheduledSequence], list[_ScheduledSequence]]:
        waiting = sorted(
            (
                runtime
                for runtime in self._active.values()
                if runtime.state.phase in {SequencePhase.QUEUED, SequencePhase.PREFILL}
            ),
            key=lambda runtime: runtime.state.created_at,
        )
        running = sorted(
            (
                runtime
                for runtime in self._active.values()
                if runtime.state.phase == SequencePhase.DECODE
            ),
            key=lambda runtime: runtime.state.created_at,
        )
        return waiting, running

    def _decision(self) -> TokenSchedulingDecision:
        waiting, running = self._sets()
        resources = TokenResourceSnapshot(
            self.max_active_sequences - len(self._active),
            self.kv_allocator.snapshot().free_blocks,
            self.max_batch_size,
        )
        decision = self.policy.select(
            [runtime.state for runtime in waiting],
            [runtime.state for runtime in running],
            resources,
        )
        if decision.decode_ids:
            # Shorter caches catch up first. Once lengths match, late joiners
            # share the same physical DynamicCache batch as older sequences.
            target_cache_length = min(
                self._active[request_id].cache_tokens
                for request_id in decision.decode_ids
            )
            compatible = tuple(
                request_id
                for request_id in decision.decode_ids
                if self._active[request_id].cache_tokens == target_cache_length
            )
            return TokenSchedulingDecision(
                decode_ids=compatible,
                reason=decision.reason,
            )
        return decision

    async def _scheduler_loop(self) -> None:
        while not self._closed or self._active:
            await self._finish_interrupted()
            decision = self._decision()
            if decision.prefill_ids:
                await self._run_prefill(decision.prefill_ids)
            elif decision.decode_ids:
                await self._run_decode(decision.decode_ids)
            else:
                self._wake.clear()
                if self._closed and not self._active:
                    break
                await self._wake.wait()

    async def _finish_interrupted(self) -> None:
        for runtime in tuple(self._active.values()):
            reason = self._check_interruption(runtime.state)
            if reason is not None:
                await self._complete(runtime, reason)

    async def _run_prefill(self, request_ids: tuple[str, ...]) -> None:
        batch: list[_ScheduledSequence] = []
        for request_id in request_ids:
            runtime = self._active.get(request_id)
            if runtime is None:
                continue
            try:
                if runtime.tokenized is None:
                    runtime.tokenized = await self.executor.run_async(
                        self.backend.tokenize, runtime.state.prompt
                    )
                    runtime.state.prompt_token_ids = runtime.tokenized.token_ids
                    runtime.state.kv_block_ids = self.kv_allocator.resize(
                        request_id,
                        len(runtime.tokenized.token_ids)
                        + runtime.state.sampling.max_new_tokens,
                    )
                runtime.state.phase = SequencePhase.PREFILL
                runtime.backend_created = True
                batch.append(runtime)
            except BaseException as exc:
                await self._fail(runtime, exc)
        if not batch:
            return
        if self.prefill_chunk_size_tokens:
            await self._run_prefill_chunks(batch)
            return
        requests = tuple(
            (runtime.state.request_id, runtime.tokenized, runtime.state.sampling)
            for runtime in batch
            if runtime.tokenized is not None
        )
        started = time.perf_counter()
        try:
            method = getattr(self.backend, "prefill_batch", None)
            if callable(method):
                outcome = await self.executor.run_async(method, requests)
            else:
                outcome = await self.executor.run_async(self._prefill_fallback, requests)
        except BaseException as exc:
            for runtime in batch:
                await self._fail(runtime, exc)
            return
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._iterations += 1
        self._prefill_batches += 1
        self._max_observed_batch_size = max(self._max_observed_batch_size, len(batch))
        try:
            eos_ids = await self._get_eos_ids()
        except BaseException as exc:
            for runtime in batch:
                await self._fail(runtime, exc)
            return
        for runtime in batch:
            failure = outcome.failures.get(runtime.state.request_id)
            if failure is not None:
                await self._fail(runtime, failure)
                continue
            runtime.state.prefill_ms += elapsed_ms
            token = outcome.tokens[runtime.state.request_id]
            runtime.cache_tokens = token.cache_tokens
            await self._accept_token(runtime, token, eos_ids)

    async def _run_prefill_chunks(self, batch: list[_ScheduledSequence]) -> None:
        method = getattr(self.backend, "prefill_chunk", None)
        if not callable(method):
            error = RuntimeError("configured backend does not support chunked prefill")
            for runtime in batch:
                await self._fail(runtime, error)
            return
        try:
            eos_ids = await self._get_eos_ids()
        except BaseException as exc:
            for runtime in batch:
                await self._fail(runtime, exc)
            return
        for runtime in batch:
            if runtime.state.request_id not in self._active or runtime.tokenized is None:
                continue
            start = runtime.prefill_offset
            end = min(
                start + self.prefill_chunk_size_tokens,
                len(runtime.tokenized.token_ids),
            )
            final = end == len(runtime.tokenized.token_ids)
            chunk = TokenizedPrompt(runtime.tokenized.token_ids[start:end])
            started = time.perf_counter()
            try:
                token = await self.executor.run_async(
                    method,
                    runtime.state.request_id,
                    chunk,
                    runtime.state.sampling,
                    final,
                )
            except BaseException as exc:
                await self._fail(runtime, exc)
                continue
            runtime.state.prefill_ms += (time.perf_counter() - started) * 1000
            runtime.prefill_offset = end
            self._iterations += 1
            self._prefill_batches += 1
            self._max_observed_batch_size = max(self._max_observed_batch_size, 1)
            if token is not None:
                runtime.cache_tokens = token.cache_tokens
                await self._accept_token(runtime, token, eos_ids)

    def _prefill_fallback(
        self,
        requests: tuple[tuple[str, TokenizedPrompt, SamplingParameters], ...],
    ) -> BatchTokenResult:
        return BatchTokenResult(
            {
                request_id: self.backend.prefill(request_id, prompt, parameters)
                for request_id, prompt, parameters in requests
            }
        )

    async def _run_decode(self, request_ids: tuple[str, ...]) -> None:
        batch = [
            self._active[request_id]
            for request_id in request_ids
            if request_id in self._active
        ]
        if not batch:
            return
        requests = tuple(
            (
                runtime.state.request_id,
                runtime.state.sampling,
                len(runtime.state.generated_token_ids),
            )
            for runtime in batch
        )
        started = time.perf_counter()
        try:
            method = getattr(self.backend, "decode_batch", None)
            if callable(method):
                outcome = await self.executor.run_async(method, requests)
            else:
                outcome = await self.executor.run_async(self._decode_fallback, requests)
        except BaseException as exc:
            for runtime in batch:
                await self._fail(runtime, exc)
            return
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._iterations += 1
        self._decode_batches += 1
        self._max_observed_batch_size = max(self._max_observed_batch_size, len(batch))
        try:
            eos_ids = await self._get_eos_ids()
        except BaseException as exc:
            for runtime in batch:
                await self._fail(runtime, exc)
            return
        for runtime in batch:
            failure = outcome.failures.get(runtime.state.request_id)
            if failure is not None:
                await self._fail(runtime, failure)
                continue
            runtime.state.decode_ms += elapsed_ms
            token = outcome.tokens[runtime.state.request_id]
            runtime.cache_tokens = token.cache_tokens
            await self._accept_token(runtime, token, eos_ids)

    def _decode_fallback(
        self,
        requests: tuple[tuple[str, SamplingParameters, int], ...],
    ) -> BatchTokenResult:
        return BatchTokenResult(
            {
                request_id: self.backend.decode(request_id, parameters, index)
                for request_id, parameters, index in requests
            }
        )

    async def _get_eos_ids(self) -> frozenset[int]:
        if self._eos_ids is None:
            self._eos_ids = await self.executor.run_async(self.backend.eos_token_ids)
        return self._eos_ids

    async def _accept_token(
        self,
        runtime: _ScheduledSequence,
        token: GeneratedToken,
        eos_ids: frozenset[int],
    ) -> None:
        sequence = runtime.state
        interrupted = self._check_interruption(sequence)
        if interrupted is not None:
            await self._complete(runtime, interrupted)
            return
        if token.token_id in eos_ids:
            await self._complete(runtime, FinishReason.EOS)
            return
        if token.token_id in sequence.sampling.stop_token_ids:
            await self._complete(runtime, FinishReason.STOP)
            return
        sequence.phase = SequencePhase.DECODE
        sequence.generated_token_ids.append(token.token_id)
        runtime.events.put_nowait(token)
        if len(sequence.generated_token_ids) >= sequence.sampling.max_new_tokens:
            await self._complete(runtime, FinishReason.LENGTH)

    async def _complete(
        self,
        runtime: _ScheduledSequence,
        reason: FinishReason,
    ) -> None:
        request_id = runtime.state.request_id
        if request_id not in self._active:
            return
        runtime.state.phase = self._phase_for(reason)
        if runtime.backend_created:
            await self.executor.run_async(self.backend.release_sequence, request_id)
            runtime.backend_created = False
        self.kv_allocator.release(request_id)
        self._active.pop(request_id, None)
        self._finished[request_id] = runtime
        runtime.events.put_nowait(
            GenerationComplete(
                reason,
                len(runtime.state.prompt_token_ids),
                len(runtime.state.generated_token_ids),
                runtime.state.prefill_ms,
                runtime.state.decode_ms,
            )
        )
        runtime.done.set()
        self._wake.set()

    async def _fail(
        self,
        runtime: _ScheduledSequence,
        error: BaseException,
    ) -> None:
        request_id = runtime.state.request_id
        if request_id not in self._active:
            return
        runtime.state.phase = SequencePhase.FAILED
        if runtime.backend_created:
            await self.executor.run_async(self.backend.release_sequence, request_id)
            runtime.backend_created = False
        self.kv_allocator.release(request_id)
        self._active.pop(request_id, None)
        self._finished[request_id] = runtime
        runtime.events.put_nowait(_SequenceFailure(error))
        runtime.done.set()
        self._wake.set()

    async def stream(
        self,
        request_id: str,
    ) -> AsyncIterator[GeneratedToken | GenerationComplete]:
        try:
            runtime = self._active.get(request_id) or self._finished[request_id]
        except KeyError as exc:
            raise KeyError(f"sequence is not reserved: {request_id}") from exc
        self._ensure_scheduler()
        try:
            while True:
                event = await runtime.events.get()
                if isinstance(event, _SequenceFailure):
                    raise event.error
                yield event
                if isinstance(event, GenerationComplete):
                    self._finished.pop(request_id, None)
                    return
        except asyncio.CancelledError:
            runtime.state.cancellation.set()
            self._wake.set()
            await asyncio.shield(runtime.done.wait())
            self._finished.pop(request_id, None)
            raise
        finally:
            if not runtime.done.is_set():
                runtime.state.cancellation.set()
                self._wake.set()
            elif runtime.events.empty():
                self._finished.pop(request_id, None)

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for runtime in self._active.values():
            runtime.state.cancellation.set()
        self._wake.set()
        if self._scheduler_task is not None:
            await self._scheduler_task
        else:
            for runtime in tuple(self._active.values()):
                await self._complete(runtime, FinishReason.CANCELLED)
        await self.executor.run_async(self.backend.shutdown)
        self.executor.shutdown()
