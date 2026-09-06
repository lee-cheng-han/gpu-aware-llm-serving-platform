# Continuous Token Scheduler

The streaming engine owns one central scheduling task. HTTP generators reserve a sequence
and consume a private event queue; they do not drive model execution. The scheduler rebuilds
waiting and running sets every iteration, admits newly arrived work, selects one workload
class, performs a device operation through the dedicated executor, and removes terminal
sequences independently.

`prefill_first` minimizes admission delay but can increase inter-token latency under prompt
bursts. `decode_first` protects active decode latency but can starve new prompts. `balanced`
alternates workload classes when both exist and is the default. `TOKEN_MAX_BATCH_SIZE` bounds
each iteration. `PREFILL_CHUNK_SIZE_TOKENS=0` performs full prefill; a positive value performs
real cache-growing prompt chunks so decode can run between chunks.

`make evaluation-token-policies` emits a deterministic, machine-readable comparison of the
three policies over identical waiting/running sets. It is labelled as decision evidence, not
hardware performance.

Hugging Face prefill batches may contain different prompt lengths using left padding. Decode
batches combine only equal physical `DynamicCache` lengths. New requests still join the
continuous scheduler, but arbitrary-length sequences cannot share one model call until a
paged-attention backend is implemented.

After a successful prefill, an enabled radix prefix cache can retain an immutable backend
snapshot. A later request restores its longest cached token prefix into an independent
sequence. An exact match skips prefill; a partial match prefills only its unmatched suffix.
This remains cloned `DynamicCache` reuse, not shared physical pages or paged attention. See
[prefix cache](prefix-cache.md).

Cancellation and deadlines are checked between every iteration and before token delivery.
An in-flight forward pass finishes, after which the affected sequence emits nothing further,
releases model cache and logical KV blocks, and reaches its terminal event. One sequence's
reported batch failure can terminate it without corrupting peers.
