# ADR 004: Strict Distributed Runtime Before Tensor Parallelism

## Context

Topology-aware placement can select a worker group, but the repository previously had no
process-group lifecycle or collective correctness boundary. Adding model sharding before
rank, device, rendezvous, failure, and cleanup behavior is explicit would make distributed
failures difficult to isolate.

## Decision

Add a typed `DistributedConfig`, deterministic tensor-shard planning, and an owned
`DistributedContext` around `torch.distributed`. NCCL mode validates CUDA, NCCL availability,
and local rank, selects the device, and refuses CPU fallback. A `torchrun` command performs
real all-reduce and broadcast checks and emits machine-readable hardware evidence.

The default remains single-rank and dependency-free. The gateway does not enable distributed
execution implicitly.

## Consequences

Process-group configuration and collective lifecycle can be tested without coupling them to
a model architecture. The project still cannot claim tensor-parallel inference, scaling, or
multi-GPU token parity until weights and forward operations are sharded and the hardware
gate is recorded.
