# ADR 003: Physical Paged KV Foundation Before Model Integration

## Context

The scheduler's fixed-block allocator provides admission and ownership accounting, while
Hugging Face `DynamicCache` owns the actual key/value tensors. Replacing model attention in
one step would mix storage correctness, model-specific tensor layouts, kernel correctness,
and CUDA performance claims.

## Decision

Introduce a PyTorch-backed page pool with a fixed physical layout, explicit per-sequence
block tables, capacity-checked append, different-length batch gather, deterministic release,
and a correctness-first attention path using PyTorch scaled-dot-product attention. CUDA is
strict: requesting it without an available device fails rather than falling back to CPU.

Keep this component outside the default Hugging Face backend until a gated integration test
proves token parity for every supported model architecture. `DynamicCache` therefore remains
the production streaming path during this step.

## Consequences

The repository now has real physical K/V tensor pages and executable block-table attention
semantics that can be tested independently. It does not yet claim end-to-end PagedAttention,
reduced fragmentation, a fused kernel, or a performance improvement. Materializing a padded
batch copies pages and is a reference implementation, not the final kernel.
