from __future__ import annotations

from typing import Protocol

from inference.token_types import SamplingParameters


class TokenSampler(Protocol):
    def select(self, logits: object, parameters: SamplingParameters) -> int: ...


class TorchTokenSampler:
    """Greedy or nucleus sampling, independent of scheduling decisions."""

    def select(self, logits: object, parameters: SamplingParameters) -> int:
        import torch

        if not isinstance(logits, torch.Tensor):
            raise TypeError("PyTorch sampler requires tensor logits")
        scores = logits.float()
        if parameters.temperature == 0:
            return int(torch.argmax(scores, dim=-1).item())
        probabilities = torch.softmax(scores / parameters.temperature, dim=-1)
        if parameters.top_p < 1:
            sorted_probabilities, sorted_indices = torch.sort(
                probabilities, descending=True, dim=-1
            )
            cumulative = torch.cumsum(sorted_probabilities, dim=-1)
            remove = cumulative - sorted_probabilities >= parameters.top_p
            sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
            sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
            sampled_index = torch.multinomial(sorted_probabilities, num_samples=1)
            return int(sorted_indices.gather(-1, sampled_index).item())
        return int(torch.multinomial(probabilities, num_samples=1).item())
