import pytest

from inference.sampling import TorchTokenSampler
from inference.token_types import SamplingParameters

torch = pytest.importorskip("torch")


def test_greedy_sampling_selects_largest_logit_and_ignores_top_p():
    sampler = TorchTokenSampler()
    logits = torch.tensor([[0.1, 3.0, 1.0]])

    assert sampler.select(logits, SamplingParameters(1, 0, 0.1)) == 1


def test_top_p_sampling_never_selects_filtered_tail(monkeypatch):
    sampler = TorchTokenSampler()
    logits = torch.tensor([[10.0, 1.0, 0.0]])

    for _ in range(20):
        assert sampler.select(logits, SamplingParameters(1, 1, 0.5)) == 0
