import time
from threading import Event, Lock, Thread

from inference.worker import GenerationResult, InferenceWorker, StreamChunk


def test_batch_and_stream_share_one_execution_lock():
    class ControlledWorker(InferenceWorker):
        def __init__(self):
            super().__init__("fake")
            self.active = 0
            self.peak = 0
            self.guard = Lock()
            self.entered = Event()

        def _enter(self):
            with self.guard:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.entered.set()
            time.sleep(.02)
            with self.guard:
                self.active -= 1

        def _generate_batch(self, prompts, max_new_tokens, temperature):
            self._enter()
            return [GenerationResult("x", 1, 1)]

        def _stream_unlocked(self, prompt, max_new_tokens, temperature):
            self._enter()
            yield StreamChunk("x", 1)

    worker = ControlledWorker()
    batch = Thread(target=worker.generate_batch, args=(["x"], 1, 0))
    stream = Thread(target=lambda: list(worker.stream("x", 1, 0)))
    batch.start()
    worker.entered.wait()
    stream.start()
    batch.join()
    stream.join()
    assert worker.peak == 1
