# Atomic Placement Reservations

Worker heartbeats are snapshots, so a scheduler can select capacity that another handoff has
already claimed but has not yet published. `AtomicPlacementReservations` closes that local
race for schedulers sharing the reservation book. Selection and reservation occur under one
lock; subsequent decisions see pending queue depth and cold-model memory. Reservations are
released after successful enqueue or every failed handoff path.

Cold memory for the same model/worker pair is counted once, matching coalesced model loading.
Reservation metrics report active handoffs, reserved model bytes, and pending demand by
model. The demand-aware policy normally prefers warm replicas but requests a memory-safe
cold replica when pending demand exceeds the configured requests-per-replica threshold.
Every decision exposes demand, current/desired replicas, scale-out intent, load, and score
terms.

The local control-plane pipeline uses this path. The atomicity boundary is process-local;
multi-controller deployments still require a shared transactional reservation adapter.
