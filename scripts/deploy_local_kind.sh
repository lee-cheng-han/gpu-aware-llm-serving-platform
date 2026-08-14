#!/usr/bin/env sh
set -eu

expected_context="kind-llm-inference"
current_context="$(kubectl config current-context)"
if [ "$current_context" != "$expected_context" ]; then
  echo "Refusing deployment: expected Kubernetes context $expected_context, got $current_context" >&2
  echo "Create/select the documented local kind cluster before deploying." >&2
  exit 1
fi

kubectl apply -f k8s/
