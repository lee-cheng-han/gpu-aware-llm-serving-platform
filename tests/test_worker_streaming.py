import sys
from queue import Queue
from types import SimpleNamespace

import pytest

from inference.worker import InferenceWorker


class FakeValue:
    def __init__(self, size):
        self.size = size

    def numel(self):
        return self.size


class FakeTextIteratorStreamer:
    def __init__(self, _tokenizer, skip_prompt, skip_special_tokens):
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.next_tokens_are_prompt = True
        self.values = Queue()

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        self.values.put(" token")

    def end(self):
        self.values.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.values.get()
        if value is None:
            raise StopIteration
        return value


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt, return_tensors):
        return {"input_ids": prompt}


def worker_with(model):
    worker = InferenceWorker("fake")
    worker.model = model
    worker.tokenizer = FakeTokenizer()
    return worker


def test_stream_counts_model_tokens(monkeypatch):
    class Model:
        def generate(self, **kwargs):
            streamer = kwargs["streamer"]
            streamer.put(FakeValue(3))
            streamer.put(FakeValue(1))
            streamer.put(FakeValue(1))
            streamer.end()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TextIteratorStreamer=FakeTextIteratorStreamer),
    )
    chunks = list(worker_with(Model()).stream("prompt", 2, 0))
    assert [chunk.text for chunk in chunks] == [" token", " token"]
    assert chunks[-1].output_tokens == 2


def test_stream_model_failure_wakes_consumer(monkeypatch):
    class Model:
        def generate(self, **kwargs):
            raise RuntimeError("model failed")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TextIteratorStreamer=FakeTextIteratorStreamer),
    )
    with pytest.raises(RuntimeError, match="streaming model generation failed"):
        list(worker_with(Model()).stream("prompt", 2, 0))
