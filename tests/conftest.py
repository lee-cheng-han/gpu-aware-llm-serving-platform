from dataclasses import dataclass

import pytest

from apps.gateway.config import Settings
from inference.worker import StreamChunk


@dataclass
class FakeResult:
    text: str
    input_tokens: int
    output_tokens: int
    tokenization_ms: float = 1
    generation_ms: float = 2
    decoding_ms: float = 0.5


class FakeWorker:
    def __init__(self, context_window=1024):
        self.batch_calls: list[list[str]] = []
        self.temperatures: list[float] = []
        self.model_name = "fake-model"
        self.load_error = None
        self.is_ready = False
        self._context_window = context_window
        self.run_inline_for_tests = True
        self.stream_inline_for_tests = True

    def warmup(self):
        self.is_ready = True

    def context_window_tokens(self):
        return self._context_window

    def count_prompt_tokens(self, prompt):
        return len(prompt.split())

    def generate_one(self, prompt, max_new_tokens, temperature):
        self.is_ready = True
        self.temperatures.append(temperature)
        self.batch_calls.append([prompt])
        return FakeResult(" generated", len(prompt.split()), max_new_tokens)

    def generate_batch(self, prompts, max_new_tokens, temperature):
        self.is_ready = True
        self.temperatures.append(temperature)
        self.batch_calls.append(prompts)
        return [FakeResult(" generated", len(p.split()), max_new_tokens) for p in prompts]

    def stream(self, prompt, max_new_tokens, temperature):
        yield StreamChunk(" generated", 2)


@pytest.fixture
def settings():
    return Settings(max_wait_ms=10, request_timeout_seconds=1)
