# Local kind Deployment

Kubernetes is deployment plumbing in v1. The deployment deliberately has one pod and one
replica so there is one queue, scheduler, metrics store, and model.

```bash
kind create cluster --name llm-inference
docker build -t gpu-aware-llm-serving-platform:local .
kind load docker-image gpu-aware-llm-serving-platform:local --name llm-inference
kubectl apply -f k8s/
kubectl port-forward -n llm-inference svc/llm-inference 8000:8000
curl http://localhost:8000/health
```

The first generation request downloads the configured model, so mount a Hugging Face
cache for repeated pod recreation if desired. `/health` only checks process health; it
does not force model warm-up.
