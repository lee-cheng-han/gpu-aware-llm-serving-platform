# Topology-Aware Placement

Workers report optional node, GPU index, NUMA node, PCIe-root, and fabric identifiers. The
topology discovery layer converts that inventory into an undirected graph. Link categories
use conservative default costs in this order: proven CUDA peer access, common PCIe root,
common host, common fabric, and general network.

`TopologyAwareWorkerGroupPolicy` first applies the existing health, runtime, model, memory,
and deadline eligibility checks. It then evaluates candidate worker groups using pairwise
shortest-path cost plus a configurable queue/load term. Stable worker-ID ordering makes ties
deterministic, and workers absent from the graph are rejected with `topology_missing`.

The graph deliberately does not label a connection as NVLink or attach a bandwidth unless
that fact was measured or explicitly supplied. This component selects groups only; it does
not initialize `torch.distributed`, execute NCCL collectives, or provide tensor parallelism.
