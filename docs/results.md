# Results

## Measured CPU token-loop proof

On 2026-08-23, the pinned SmolLM2-135M-Instruct revision was tested on an Intel Core Ultra 7
155H CPU under WSL2 with Python 3.12.14, PyTorch 2.7.0+cpu, Transformers 4.51.3, and
`float32`. The optional integration test proved that greedy manual-loop token IDs match
Hugging Face `generate()` and that the dynamic KV cache is created at prefill and grows at
decode. A second real-model integration test exercised the existing SSE URL end to end and
verified non-emission plus correct terminal usage for both EOS and configured stop tokens.

The correctness-oriented benchmark used one warmup, two measured trials, one fixed prompt,
and four output tokens. Whole-request `generate()` averaged 262.4 ms and 15.59 tokens/s;
the manual loop averaged 296.0 ms end to end, 107.5 ms TTFT, 62.8 ms per subsequent decode
step, and 13.53 tokens/s. The manual loop was about 13% slower in this very small sample; no
improvement is claimed. Legacy TTFT is unavailable because `generate()` returns only at the end.
The complete workload and raw trials are in
[`docs/evidence/token-backend-cpu.json`](evidence/token-backend-cpu.json).

No GPU hardware measurement has been run. GPU validation remains pending and no CPU result
should be generalized to CUDA.

The versioned deterministic baseline is available at
[`benchmark/results/simulated/heterogeneous/report.md`](../benchmark/results/simulated/heterogeneous/report.md).
It is a control-plane regression fixture, not a GPU-aware platform measurement.
