import time
from dataclasses import dataclass
from threading import Lock, Thread

from inference.model_loader import load_model


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    tokenization_ms: float = 0
    generation_ms: float = 0
    decoding_ms: float = 0


@dataclass
class StreamChunk:
    text: str
    output_tokens: int


class InferenceWorker:
    """Lazy CPU worker. Scheduler calls it through asyncio.to_thread."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = self.model = None
        self._load_lock = Lock()
        self._execution_lock = Lock()
        self.load_error: str | None = None

    def _ensure_loaded(self):
        if self.model is None:
            with self._load_lock:
                if self.model is None:
                    try:
                        self.tokenizer, self.model = load_model(self.model_name)
                        self.load_error = None
                    except Exception as exc:
                        self.load_error = str(exc)
                        raise

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def warmup(self) -> None:
        self._ensure_loaded()

    def context_window_tokens(self) -> int:
        self._ensure_loaded()
        candidates = [
            getattr(self.model.config, "max_position_embeddings", None),
            getattr(self.tokenizer, "model_max_length", None),
        ]
        # Some tokenizers use an enormous sentinel for "unknown".
        valid = [int(value) for value in candidates if value and int(value) < 1_000_000]
        if not valid:
            raise RuntimeError("model does not expose a usable context window")
        return min(valid)

    def count_prompt_tokens(self, prompt: str) -> int:
        self._ensure_loaded()
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def generate_one(self, prompt: str, max_new_tokens: int, temperature: float) -> GenerationResult:
        return self.generate_batch([prompt], max_new_tokens, temperature)[0]

    def generate_batch(self, prompts: list[str], max_new_tokens: int, temperature: float) -> list[GenerationResult]:
        with self._execution_lock:
            return self._generate_batch(prompts, max_new_tokens, temperature)

    def _generate_batch(self, prompts: list[str], max_new_tokens: int, temperature: float) -> list[GenerationResult]:
        import torch
        self._ensure_loaded()
        tokenization_started = time.perf_counter()
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature
        generation_started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(**encoded, **generation_args)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        prompt_width = encoded["input_ids"].shape[1]
        results = []
        for i, output in enumerate(outputs):
            decoding_started = time.perf_counter()
            raw_new_ids = output[prompt_width:].tolist()
            meaningful_ids = []
            for token_id in raw_new_ids:
                meaningful_ids.append(token_id)
                if token_id == self.tokenizer.eos_token_id:
                    break
            new_ids = meaningful_ids
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(GenerationResult(
                text=text,
                input_tokens=int(input_lengths[i]),
                output_tokens=int(len(new_ids)),
                tokenization_ms=tokenization_ms,
                generation_ms=generation_ms,
                decoding_ms=(time.perf_counter() - decoding_started) * 1000,
            ))
        return results

    def stream(self, prompt: str, max_new_tokens: int, temperature: float):
        with self._execution_lock:
            yield from self._stream_unlocked(prompt, max_new_tokens, temperature)

    def _stream_unlocked(self, prompt: str, max_new_tokens: int, temperature: float):
        from transformers import TextIteratorStreamer

        class CountingStreamer(TextIteratorStreamer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.generated_tokens = 0

            def put(self, value):
                is_prompt = self.skip_prompt and self.next_tokens_are_prompt
                if not is_prompt:
                    self.generated_tokens += int(value.numel())
                return super().put(value)

        self._ensure_loaded()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        streamer = CountingStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            **encoded, streamer=streamer, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            kwargs["temperature"] = temperature
        failures: list[BaseException] = []

        def generate() -> None:
            try:
                self.model.generate(**kwargs)
            except BaseException as exc:
                failures.append(exc)
                # generate() normally ends the streamer itself. On failure it
                # may not, so wake the consumer explicitly.
                streamer.end()

        thread = Thread(target=generate, daemon=True)
        thread.start()
        try:
            for text in streamer:
                yield StreamChunk(text=text, output_tokens=streamer.generated_tokens)
        finally:
            thread.join()
        if failures:
            raise RuntimeError("streaming model generation failed") from failures[0]
