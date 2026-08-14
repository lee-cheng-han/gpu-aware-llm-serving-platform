# Local kind Deployment

> These manifests are intended for the local `kind-llm-inference` cluster. Kubernetes YAML
> cannot determine whether the active context is a billable managed cluster. Use the guarded
> `make deploy-kind` command instead of applying the directory directly.

Kubernetes is deployment plumbing in v1. The deployment deliberately has one pod and one
replica so there is one queue, scheduler, metrics store, and model.

```bash
kind create cluster --name llm-inference
docker build -t gpu-aware-llm-serving-platform:local .
kind load docker-image gpu-aware-llm-serving-platform:local --name llm-inference
make deploy-kind
kubectl port-forward -n llm-inference svc/llm-inference 8000:8000
curl http://localhost:8000/health
```

The pod runs as UID/GID 10001 with all Linux capabilities dropped, privilege escalation
disabled, a read-only root filesystem, and writable size-bounded temporary storage. The
model cache is an `emptyDir`; replace it with a suitable persistent volume when repeated
downloads are undesirable. The startup probe allows up to five minutes for initial model
download and warm-up. `/health` checks the process while `/ready` requires a loaded model.

The default network policy accepts port 8000 only from the same namespace and permits DNS
plus outbound HTTPS for model downloads. Adapt its ingress selector if an ingress controller
or monitoring agent runs elsewhere. The one-replica disruption budget intentionally blocks
voluntary eviction because this example has no highly available request-state deployment.
