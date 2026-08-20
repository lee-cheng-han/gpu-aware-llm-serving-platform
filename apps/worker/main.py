from __future__ import annotations

import os

from apps.worker.api import create_worker_http_app
from apps.worker.factory import create_local_simulated_worker


def create_app():
    worker_id = os.getenv("WORKER_ID", "local-sim-worker")
    auth_token = os.getenv("WORKER_AUTH_TOKEN", "")
    if not auth_token:
        raise RuntimeError("WORKER_AUTH_TOKEN is required")
    try:
        tokens_per_second = float(os.getenv("SIMULATED_TOKENS_PER_SECOND", "10000"))
    except ValueError as exc:
        raise RuntimeError("SIMULATED_TOKENS_PER_SECOND must be numeric") from exc
    worker, model = create_local_simulated_worker(worker_id, tokens_per_second)
    return create_worker_http_app(worker, [model], auth_token)


app = create_app()
