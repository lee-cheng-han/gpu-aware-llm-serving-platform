# ADR 001: Explicit Hugging Face Prefill and Decode

## Context

The compatibility worker performs real inference through complete `model.generate()` calls.
That proves model integration but hides prefill, decode, cache lifetime, and token-level
cancellation—the boundaries required by continuous batching and paged KV management.

## Alternatives

- Keep `generate()`: lowest complexity, but cannot expose scheduler-controlled iterations.
- Adopt vLLM/TGI immediately: capable, but outsources the systems mechanisms this project is
  intended to demonstrate.
- Build a custom CUDA runtime now: premature before correctness and profiling evidence.

## Decision

Add a typed Hugging Face backend that runs a real prompt prefill followed by one-token decode
steps with `use_cache=True` and `DynamicCache`. One dedicated executor thread serializes all
operations for a backend. The async streaming engine wraps those futures, while the existing
static non-streaming path remains compatible.

Sampling is a separate component. Every sequence has an explicit phase and terminal reason.
Cancellation is checked around each device step; an in-flight PyTorch call may finish, but
no token is emitted afterward and sequence cache state is always released.

## Tradeoffs

This produces genuine KV-cached autoregressive inference and a clean scheduling boundary,
but it is single-device and not continuous batching. Framework-managed dynamic cache is not
paged KV memory. The CPU path can be validated locally; CUDA support remains unverified
until the GPU-gated evidence suite runs on identified hardware.
