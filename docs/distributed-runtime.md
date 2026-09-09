# Distributed Runtime Foundation

`DistributedConfig` reads standard `torchrun` rank variables and validates world size,
global rank, local rank, backend, rendezvous method, and timeout. `DistributedContext` owns
process-group setup, all-reduce, broadcast, object gather, barriers, status, and idempotent
shutdown.

NCCL mode is strict. It verifies CUDA, NCCL support, and the local device index, selects that
device, and fails without changing to Gloo or CPU. A single-rank context remains a
dependency-free identity path for local development. `balanced_shard` provides deterministic
uneven tensor-dimension partitioning for the later model-sharding layer.

Run the real two-process collective check on an owned or free machine with at least two
CUDA devices:

```bash
make test-distributed-gpu
```

The runner checks all-reduce and broadcast results and can write a machine-readable report
with `--output`. This layer does not yet shard SmolLM2 weights, combine tensor-parallel
logits, or route gateway requests through a distributed worker group. Those are the remaining
model-execution steps; no multi-GPU result is claimed until the gated report passes.
