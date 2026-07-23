from dataclasses import dataclass
from threading import Lock
from threading import Thread

from inference.model_loader import load_model


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int


class InferenceWorker:
    """Lazy CPU worker. Scheduler calls it through asyncio.to_thread."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = self.model = None
        self._load_lock = Lock()
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
        import torch
        self._ensure_loaded()
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature
        with torch.no_grad():
            outputs = self.model.generate(**encoded, **generation_args)
        prompt_width = encoded["input_ids"].shape[1]
        results = []
        for i, output in enumerate(outputs):
            raw_new_ids = output[prompt_width:].tolist()
            meaningful_ids = []
            for token_id in raw_new_ids:
                meaningful_ids.append(token_id)
                if token_id == self.tokenizer.eos_token_id:
                    break
            new_ids = meaningful_ids
            results.append(GenerationResult(
                self.tokenizer.decode(new_ids, skip_special_tokens=True),
                int(input_lengths[i]), int(len(new_ids)),
            ))
        return results

    def stream(self, prompt: str, max_new_tokens: int, temperature: float):
        from transformers import TextIteratorStreamer
        self._ensure_loaded()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            **encoded, streamer=streamer, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            kwargs["temperature"] = temperature
        thread = Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        thread.start()
        yield from streamer
        thread.join()
