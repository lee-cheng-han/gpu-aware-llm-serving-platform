from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from inference.backend import BatchTokenResult
from inference.sampling import TokenSampler, TorchTokenSampler
from inference.token_types import GeneratedToken, SamplingParameters, TokenizedPrompt
from inference.worker import InferenceWorker
from runtime.base import RuntimeCapacity
from serving_platform.domain import DeviceType


@dataclass
class _SequenceCache:
    cache: Any
    attention_mask: Any
    cache_position: Any
    last_token_id: int


class HuggingFaceTokenBackend:
    """Single-device, real prefill/decode backend over a shared compatibility worker."""

    def __init__(
        self,
        worker: InferenceWorker,
        sampler: TokenSampler | None = None,
    ):
        self.worker = worker
        self.sampler = sampler or TorchTokenSampler()
        self._sequences: dict[str, _SequenceCache] = {}
        self._stopped = False

    def _require_active(self) -> None:
        if self._stopped:
            raise RuntimeError("inference backend is shut down")

    def load_model(self) -> None:
        self._require_active()
        if self.worker.device.startswith("cuda"):
            torch = import_module("torch")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA backend requested but CUDA is unavailable")
        self.worker.warmup()

    def unload_model(self) -> None:
        if self._sequences:
            raise RuntimeError("cannot unload a model with active sequences")
        self.worker.unload()

    def tokenize(self, prompt: str) -> TokenizedPrompt:
        self.load_model()
        with self.worker._execution_lock:
            tokenizer = self.worker.tokenizer
            if tokenizer is None:
                raise RuntimeError("tokenizer is unavailable after model load")
            token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        return TokenizedPrompt(tuple(int(token_id) for token_id in token_ids))

    def vocabulary_size(self) -> int:
        self.load_model()
        with self.worker._execution_lock:
            tokenizer = self.worker.tokenizer
            if tokenizer is None:
                raise RuntimeError("tokenizer is unavailable after model load")
            return int(len(tokenizer))

    def eos_token_ids(self) -> frozenset[int]:
        self.load_model()
        with self.worker._execution_lock:
            model = self.worker.model
            tokenizer = self.worker.tokenizer
            if model is None or tokenizer is None:
                raise RuntimeError("model components are unavailable after model load")
            configured = getattr(model.generation_config, "eos_token_id", None)
            if configured is None:
                configured = tokenizer.eos_token_id
        values = configured if isinstance(configured, (list, tuple)) else [configured]
        return frozenset(int(value) for value in values if value is not None)

    def _token(self, logits: object, parameters: SamplingParameters, index: int) -> int:
        return self.sampler.select(logits, parameters)

    def _decode_text(self, token_id: int) -> str:
        tokenizer = self.worker.tokenizer
        if tokenizer is None:
            raise RuntimeError("tokenizer is unavailable during decode")
        return str(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

    @staticmethod
    def _cache_length(cache: object) -> int:
        get_length = getattr(cache, "get_seq_length", None)
        return int(get_length()) if get_length is not None else 0

    def prefill(
        self,
        sequence_id: str,
        prompt: TokenizedPrompt,
        parameters: SamplingParameters,
    ) -> GeneratedToken:
        self._require_active()
        if sequence_id in self._sequences:
            raise ValueError(f"sequence already exists: {sequence_id}")
        torch = import_module("torch")
        dynamic_cache = import_module("transformers").DynamicCache

        self.load_model()
        with self.worker._execution_lock, torch.inference_mode():
            model = self.worker.model
            if model is None:
                raise RuntimeError("model is unavailable after model load")
            input_ids = torch.tensor(
                [prompt.token_ids], dtype=torch.long, device=self.worker.device
            )
            attention_mask = torch.ones_like(input_ids)
            cache_position = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=self.worker.device
            )
            cache = dynamic_cache()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
            cache = outputs.past_key_values
            token_id = self._token(outputs.logits[:, -1, :], parameters, 0)
            self._sequences[sequence_id] = _SequenceCache(
                cache=cache,
                attention_mask=attention_mask,
                cache_position=cache_position[-1:] + 1,
                last_token_id=token_id,
            )
            return GeneratedToken(
                token_id,
                self._decode_text(token_id),
                0,
                self._cache_length(cache),
            )

    def prefill_batch(
        self,
        requests: tuple[tuple[str, TokenizedPrompt, SamplingParameters], ...],
    ) -> BatchTokenResult:
        if not requests:
            return BatchTokenResult({})
        if len(requests) == 1:
            sequence_id, prompt, parameters = requests[0]
            return BatchTokenResult(
                {sequence_id: self.prefill(sequence_id, prompt, parameters)}
            )
        self._require_active()
        if len({sequence_id for sequence_id, _, _ in requests}) != len(requests):
            raise ValueError("prefill batch contains duplicate sequence identifiers")
        if any(sequence_id in self._sequences for sequence_id, _, _ in requests):
            raise ValueError("prefill batch contains an existing sequence")
        torch = import_module("torch")
        dynamic_cache = import_module("transformers").DynamicCache
        self.load_model()
        with self.worker._execution_lock, torch.inference_mode():
            model = self.worker.model
            tokenizer = self.worker.tokenizer
            if model is None or tokenizer is None:
                raise RuntimeError("model components are unavailable after model load")
            width = max(len(prompt.token_ids) for _, prompt, _ in requests)
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                raise RuntimeError("tokenizer does not define a padding token")
            rows = []
            masks = []
            for _, prompt, _ in requests:
                padding = width - len(prompt.token_ids)
                rows.append([pad_id] * padding + list(prompt.token_ids))
                masks.append([0] * padding + [1] * len(prompt.token_ids))
            input_ids = torch.tensor(rows, dtype=torch.long, device=self.worker.device)
            attention_mask = torch.tensor(
                masks, dtype=torch.long, device=self.worker.device
            )
            cache_position = torch.arange(width, dtype=torch.long, device=self.worker.device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=dynamic_cache(),
                cache_position=cache_position,
                use_cache=True,
            )
            caches = outputs.past_key_values.batch_split(len(requests), 1)
            result: dict[str, GeneratedToken] = {}
            for row, ((sequence_id, _, parameters), cache) in enumerate(
                zip(requests, caches, strict=True)
            ):
                token_id = self._token(outputs.logits[row : row + 1, -1, :], parameters, 0)
                self._sequences[sequence_id] = _SequenceCache(
                    cache=cache,
                    attention_mask=attention_mask[row : row + 1],
                    cache_position=cache_position[-1:] + 1,
                    last_token_id=token_id,
                )
                result[sequence_id] = GeneratedToken(
                    token_id,
                    self._decode_text(token_id),
                    0,
                    self._cache_length(cache),
                )
            return BatchTokenResult(result)

    def prefill_chunk(
        self,
        sequence_id: str,
        prompt: TokenizedPrompt,
        parameters: SamplingParameters,
        final: bool,
    ) -> GeneratedToken | None:
        """Advance a prompt cache without sampling until the final prompt chunk."""
        self._require_active()
        torch = import_module("torch")
        dynamic_cache = import_module("transformers").DynamicCache
        self.load_model()
        with self.worker._execution_lock, torch.inference_mode():
            model = self.worker.model
            if model is None:
                raise RuntimeError("model is unavailable after model load")
            existing = self._sequences.get(sequence_id)
            input_ids = torch.tensor(
                [prompt.token_ids], dtype=torch.long, device=self.worker.device
            )
            if existing is None:
                attention_mask = torch.ones_like(input_ids)
                start = 0
                cache = dynamic_cache()
            else:
                attention_mask = torch.cat(
                    (existing.attention_mask, torch.ones_like(input_ids)), dim=-1
                )
                start = self._cache_length(existing.cache)
                cache = existing.cache
            cache_position = torch.arange(
                start,
                start + input_ids.shape[1],
                dtype=torch.long,
                device=self.worker.device,
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
            cache = outputs.past_key_values
            sequence = _SequenceCache(
                cache=cache,
                attention_mask=attention_mask,
                cache_position=cache_position[-1:] + 1,
                last_token_id=-1,
            )
            self._sequences[sequence_id] = sequence
            if not final:
                return None
            token_id = self._token(outputs.logits[:, -1, :], parameters, 0)
            sequence.last_token_id = token_id
            return GeneratedToken(
                token_id,
                self._decode_text(token_id),
                0,
                self._cache_length(cache),
            )

    def decode(
        self,
        sequence_id: str,
        parameters: SamplingParameters,
        index: int,
    ) -> GeneratedToken:
        self._require_active()
        try:
            sequence = self._sequences[sequence_id]
        except KeyError as exc:
            raise KeyError(f"sequence does not exist: {sequence_id}") from exc
        torch = import_module("torch")

        with self.worker._execution_lock, torch.inference_mode():
            model = self.worker.model
            if model is None:
                raise RuntimeError("model is unavailable during decode")
            input_ids = torch.tensor(
                [[sequence.last_token_id]],
                dtype=torch.long,
                device=self.worker.device,
            )
            sequence.attention_mask = torch.cat(
                (
                    sequence.attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=self.worker.device),
                ),
                dim=-1,
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=sequence.attention_mask,
                past_key_values=sequence.cache,
                cache_position=sequence.cache_position,
                use_cache=True,
            )
            sequence.cache = outputs.past_key_values
            sequence.cache_position = sequence.cache_position + 1
            token_id = self._token(outputs.logits[:, -1, :], parameters, index)
            sequence.last_token_id = token_id
            return GeneratedToken(
                token_id,
                self._decode_text(token_id),
                index,
                self._cache_length(sequence.cache),
            )

    def decode_batch(
        self,
        requests: tuple[tuple[str, SamplingParameters, int], ...],
    ) -> BatchTokenResult:
        if not requests:
            return BatchTokenResult({})
        if len(requests) == 1:
            sequence_id, parameters, index = requests[0]
            return BatchTokenResult(
                {sequence_id: self.decode(sequence_id, parameters, index)}
            )
        torch = import_module("torch")
        dynamic_cache = import_module("transformers").DynamicCache
        sequences = [self._sequences[sequence_id] for sequence_id, _, _ in requests]
        cache_lengths = {self._cache_length(sequence.cache) for sequence in sequences}
        if len(cache_lengths) != 1:
            raise ValueError("decode batches require equal physical cache lengths")
        with self.worker._execution_lock, torch.inference_mode():
            model = self.worker.model
            if model is None:
                raise RuntimeError("model is unavailable during decode")
            input_ids = torch.tensor(
                [[sequence.last_token_id] for sequence in sequences],
                dtype=torch.long,
                device=self.worker.device,
            )
            attention_mask = torch.cat(
                [
                    torch.cat(
                        (
                            sequence.attention_mask,
                            torch.ones(
                                (1, 1), dtype=torch.long, device=self.worker.device
                            ),
                        ),
                        dim=-1,
                    )
                    for sequence in sequences
                ],
                dim=0,
            )
            cache = dynamic_cache.from_batch_splits(
                [sequence.cache for sequence in sequences]
            )
            cache_position = sequences[0].cache_position
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
            caches = outputs.past_key_values.batch_split(len(requests), 1)
            result: dict[str, GeneratedToken] = {}
            for row, ((sequence_id, parameters, index), sequence, split_cache) in enumerate(
                zip(requests, sequences, caches, strict=True)
            ):
                token_id = self._token(
                    outputs.logits[row : row + 1, -1, :], parameters, index
                )
                sequence.cache = split_cache
                sequence.attention_mask = attention_mask[row : row + 1]
                sequence.cache_position = sequence.cache_position + 1
                sequence.last_token_id = token_id
                result[sequence_id] = GeneratedToken(
                    token_id,
                    self._decode_text(token_id),
                    index,
                    self._cache_length(split_cache),
                )
            return BatchTokenResult(result)

    def release_sequence(self, sequence_id: str) -> None:
        self._sequences.pop(sequence_id, None)

    def active_sequence_count(self) -> int:
        return len(self._sequences)

    def capacity(self) -> RuntimeCapacity:
        if not self.worker.device.startswith("cuda"):
            return RuntimeCapacity(DeviceType.CPU, "cpu", None, None)
        torch = import_module("torch")
        index = int(self.worker.device.partition(":")[2] or 0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        return RuntimeCapacity(
            DeviceType.CUDA,
            torch.cuda.get_device_name(index),
            int(total_bytes),
            int(free_bytes),
            int(torch.cuda.memory_allocated(index)),
            int(torch.cuda.memory_reserved(index)),
        )

    def shutdown(self) -> None:
        for sequence_id in tuple(self._sequences):
            self.release_sequence(sequence_id)
        self.worker.unload()
        self._stopped = True
