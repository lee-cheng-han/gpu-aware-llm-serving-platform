from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from apps.control_plane import ControlPlane, GlobalScheduler, WorkerDirectory
from apps.worker.factory import local_simulated_model
from apps.worker.transport import HttpWorkerClient
from serving_platform.domain import RequestRecord, RequestState
from serving_platform.registry import InMemoryModelRegistry, InMemoryWorkerRegistry
from serving_platform.routing import ModelResidencyAwarePolicy


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@contextmanager
def worker_process(worker_id: str, port: int, token: str) -> Iterator[subprocess.Popen[bytes]]:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(root),
            "WORKER_ID": worker_id,
            "WORKER_AUTH_TOKEN": token,
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "apps.worker.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def await_worker(client: HttpWorkerClient, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _, stderr = process.communicate()
            raise AssertionError(f"worker exited during startup: {stderr.decode()}")
        try:
            client.health()
            return
        except RuntimeError:
            time.sleep(0.05)
    raise AssertionError("worker did not become reachable")


def test_control_plane_dispatches_across_independent_worker_processes():
    token = "local-integration-secret"
    ports = (available_port(), available_port())
    registry = InMemoryWorkerRegistry()
    clients = [
        HttpWorkerClient("worker-a", f"http://127.0.0.1:{ports[0]}", token, registry),
        HttpWorkerClient("worker-b", f"http://127.0.0.1:{ports[1]}", token, registry),
    ]
    with worker_process("worker-a", ports[0], token) as first:
        with worker_process("worker-b", ports[1], token) as second:
            for client, process in zip(clients, (first, second), strict=True):
                await_worker(client, process)
                client.register()

            directory = WorkerDirectory()
            for client in clients:
                directory.add(client)
            model = local_simulated_model()
            models = InMemoryModelRegistry([model])
            control_plane = ControlPlane(
                GlobalScheduler(registry, ModelResidencyAwarePolicy(models)),
                directory,
                models,
            )
            request = RequestRecord(
                "process-request",
                "tenant",
                model.model_id,
                "explain local scheduling",
                3,
                4,
                0,
                float("inf"),
                False,
            )
            request.transition(RequestState.VALIDATED)
            request.transition(RequestState.ADMITTED)

            assignment = control_plane.dispatch(request)
            executions = clients[0].execute_batch()

            assert first.pid != second.pid != os.getpid()
            assert assignment.worker_id == "worker-a"
            assert executions[0].request_id == request.request_id
            assert executions[0].result.text.startswith("[simulated:")
            assert request.status == RequestState.COMPLETED
            assert registry.get("worker-a").resident_models == {model.model_id}
