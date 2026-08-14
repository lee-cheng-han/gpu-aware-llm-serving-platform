# Architecture decisions

## Modular monolith before distributed deployment

The gateway, control plane, and worker are separate modules with inward-facing interfaces,
but remain deployable in one process while state ownership and failure semantics mature.
The compatibility gateway is not advertised as distributed execution.

## Persistence excludes inference payloads

The Redis-compatible store keeps request identity, lifecycle, assignment, attempts, retry
reasons, and timestamps. It excludes prompts and outputs by default. Metadata restored after
a process restart is not executable without a separately designed encrypted payload store.

## Retry only when execution provably never started

Worker loss can requeue assigned work only if no running or streaming transition exists.
Potentially started work fails instead of risking duplicate inference. Retry count, reason,
and the original deadline remain authoritative.

## Weighted fairness is token-cost aware

Weighted deficit round robin allocates service by tenant weight and estimated token cost.
Priority affects order within a tenant, while aging prevents permanent starvation. The
deterministic evaluation records a normalized weighted-fairness score.

## Simulation and measurements are different products

The simulated runtime and evaluation runner validate scheduling behavior repeatably. Their
throughput is synthetic and must be labelled. Real reports require recorded hardware,
software, model revision, configuration, seed, and raw trials.

## Model identity is more than a name

Model definitions already carry revision, runtime, dtype support, context, and batch limits.
Some cache and routing paths still key by model ID; revision/dtype-complete cache identity is
tracked as remaining production work rather than hidden behind documentation.

## Dynamic batching is not continuous batching

This project forms short-window batches and runs whole Hugging Face `generate()` calls.
It does not add or remove sequences at token boundaries and does not implement PagedAttention
or a custom KV-cache manager.
