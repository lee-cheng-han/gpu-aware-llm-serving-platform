# Local Qualification Harness

Run the zero-cost, download-free qualification drills with:

```bash
make qualification
```

The command emits JSON and exits unsuccessfully when a gate fails. It uses a fixed seed to
exercise KV block allocation and release, checks prefix-cache entry/token bounds, verifies
admission overload rejection and cleanup, injects one deterministic runtime failure, proves
a later call recovers, and checks bounded Python heap growth after cleanup.

Use `--iterations`, `--max-python-heap-growth-bytes`, and `--output` through
`python -m benchmark.run_qualification` to tune a local run or retain its report. The
classification field explicitly labels the result as deterministic local Python evidence.
It does not measure CUDA allocations, GPU kernel stability, driver recovery, throughput, or
multi-process production behavior. Those claims require recorded runs on identified GPU
hardware.
