# Deployment profiles

Three profiles are intentionally distinct:

- `local-cpu.env` runs the public FastAPI gateway with the default CPU image and no paid
  inference API.
- `simulated-gpu.env` describes the deterministic evaluation runner. It is not an API
  server, never uses CUDA, and its output must remain labelled as simulated.
- `cuda.env.example` runs the compatibility gateway on one real NVIDIA device using
  `Dockerfile.cuda`. It requires an NVIDIA driver, NVIDIA Container Toolkit, and local GPU
  capacity; it does not call a paid inference API.

None of these profiles provisions infrastructure. Do not run the Kubernetes manifests on a
managed cluster or replace the local CUDA profile with a hosted GPU unless you intentionally
accept that provider's charges.

Run CPU locally:

```bash
set -a
. deploy/profiles/local-cpu.env
set +a
make run
```

Run the deterministic simulation and its gates:

```bash
python -m benchmark.run_simulated_evaluation --check
```

Build and run CUDA:

```bash
docker build -f Dockerfile.cuda -t gpu-aware-llm-serving-platform:cuda .
docker run --rm --gpus all -p 8000:8000 \
  --env-file deploy/profiles/cuda.env.example \
  gpu-aware-llm-serving-platform:cuda
```

For real-device reports, record the GPU model and memory, driver, CUDA, Python, PyTorch,
Transformers, model revision, environment profile, workload, and seed. Never compare a
simulated report with a real-device result.
