from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from serving_platform.admission.interfaces import AdmissionDecision
from serving_platform.domain import (
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    TenantLimits,
    WorkerState,
)


class RedisScriptClient(Protocol):
    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> object: ...


CONFIGURE_SCRIPT = """
-- admission_configure_v1
local state = KEYS[1]
local current = redis.call('HGET', state, 'config:fingerprint')
if not current then
  redis.call('HSET', state, 'config:fingerprint', ARGV[1])
  return 'configured'
end
if current == ARGV[1] then return 'existing' end
return 'conflict'
"""

RESERVE_SCRIPT = """
-- admission_reserve_v1
local state = KEYS[1]
local reservation = 'reservation:' .. ARGV[1]
local tenant = ARGV[2]
local tokens = tonumber(ARGV[3])
local global_limit = tonumber(ARGV[4])
local queue_limit = tonumber(ARGV[5])
local concurrency_limit = tonumber(ARGV[6])
local token_limit = tonumber(ARGV[7])
local queued_key = 'tenant:' .. tenant .. ':queued'
local running_key = 'tenant:' .. tenant .. ':running'
local tokens_key = 'tenant:' .. tenant .. ':tokens'
local global_queued = tonumber(redis.call('HGET', state, 'global:queued') or '0')
local queued = tonumber(redis.call('HGET', state, queued_key) or '0')
local running = tonumber(redis.call('HGET', state, running_key) or '0')
local reserved_tokens = tonumber(redis.call('HGET', state, tokens_key) or '0')

if redis.call('HEXISTS', state, reservation) == 1 then return 'already_admitted' end
if global_queued >= global_limit then return 'global_queue_full' end
if queued >= queue_limit then return 'tenant_queue_full' end
if running >= concurrency_limit then return 'tenant_concurrency_limit' end
if reserved_tokens + tokens > token_limit then return 'tenant_token_quota' end

redis.call('HINCRBY', state, 'global:queued', 1)
redis.call('HINCRBY', state, queued_key, 1)
redis.call('HINCRBY', state, tokens_key, tokens)
redis.call('HSET', state, reservation, cjson.encode({
  tenant=tenant,tokens=tokens,running=false,concurrency_limit=concurrency_limit
}))
return 'admitted'
"""

MARK_RUNNING_SCRIPT = """
-- admission_mark_running_v1
local state = KEYS[1]
local reservation_key = 'reservation:' .. ARGV[1]
local encoded = redis.call('HGET', state, reservation_key)
if not encoded then return 'missing' end
local reservation = cjson.decode(encoded)
if reservation.running then return 'already_running' end
local tenant = reservation.tenant
local queued_key = 'tenant:' .. tenant .. ':queued'
local running_key = 'tenant:' .. tenant .. ':running'
local running = tonumber(redis.call('HGET', state, running_key) or '0')
if running >= reservation.concurrency_limit then return 'tenant_concurrency_limit' end

redis.call('HINCRBY', state, 'global:queued', -1)
redis.call('HINCRBY', state, queued_key, -1)
redis.call('HINCRBY', state, running_key, 1)
reservation.running = true
redis.call('HSET', state, reservation_key, cjson.encode(reservation))
return 'running'
"""

RELEASE_SCRIPT = """
-- admission_release_v1
local state = KEYS[1]
local reservation_key = 'reservation:' .. ARGV[1]
local encoded = redis.call('HGET', state, reservation_key)
if not encoded then return 0 end
local reservation = cjson.decode(encoded)
local tenant = reservation.tenant
if reservation.running then
  redis.call('HINCRBY', state, 'tenant:' .. tenant .. ':running', -1)
else
  redis.call('HINCRBY', state, 'global:queued', -1)
  redis.call('HINCRBY', state, 'tenant:' .. tenant .. ':queued', -1)
end
redis.call('HINCRBY', state, 'tenant:' .. tenant .. ':tokens', -reservation.tokens)
redis.call('HDEL', state, reservation_key)
return 1
"""

SNAPSHOT_SCRIPT = """
-- admission_snapshot_v1
local state = KEYS[1]
local tenant = ARGV[1]
return {
  redis.call('HGET', state, 'tenant:' .. tenant .. ':queued') or '0',
  redis.call('HGET', state, 'tenant:' .. tenant .. ':running') or '0',
  redis.call('HGET', state, 'tenant:' .. tenant .. ':tokens') or '0'
}
"""


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class RedisAdmissionController:
    """Cross-process admission counters updated atomically by Redis Lua scripts."""

    def __init__(
        self,
        client: RedisScriptClient,
        tenants: Sequence[TenantLimits],
        global_queue_capacity: int,
        namespace: str = "default",
        clock: Callable[[], float] = time.monotonic,
    ):
        if global_queue_capacity <= 0:
            raise ValueError("global queue capacity must be positive")
        if not namespace or "{" in namespace or "}" in namespace:
            raise ValueError("admission namespace is invalid")
        self.client = client
        self.tenants = {tenant.tenant_id: tenant for tenant in tenants}
        if len(self.tenants) != len(tenants):
            raise ValueError("tenant identifiers must be unique")
        self.global_queue_capacity = global_queue_capacity
        self.state_key = f"llm:admission:{{{namespace}}}:state"
        self._clock = clock
        configuration = {
            "global_queue_capacity": global_queue_capacity,
            "tenants": [
                {
                    "tenant_id": tenant.tenant_id,
                    "max_concurrent_requests": tenant.max_concurrent_requests,
                    "max_queued_requests": tenant.max_queued_requests,
                    "token_quota": tenant.token_quota,
                }
                for tenant in sorted(tenants, key=lambda item: item.tenant_id)
            ],
        }
        fingerprint = hashlib.sha256(
            json.dumps(configuration, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        configured = _text(
            self.client.eval(CONFIGURE_SCRIPT, 1, self.state_key, fingerprint)
        )
        if configured not in {"configured", "existing"}:
            raise ValueError("shared admission configuration conflicts with existing state")

    def decide(
        self,
        request: RequestRecord,
        model: ModelDefinition | None,
        workers: Sequence[WorkerState],
    ) -> AdmissionDecision:
        tenant = self.tenants.get(request.tenant_id)
        if tenant is None:
            return AdmissionDecision(False, "unknown_tenant", "tenant is not configured")
        if model is None:
            return AdmissionDecision(False, "unsupported_model", "model is not registered")
        if request.estimated_tokens > model.max_context_tokens:
            return AdmissionDecision(False, "context_window_exceeded", "context is too large")
        if request.deadline <= self._clock():
            return AdmissionDecision(False, "deadline_exceeded", "deadline has expired")
        if not any(
            worker.health_status == HealthStatus.HEALTHY and not worker.draining
            for worker in workers
        ):
            return AdmissionDecision(False, "no_healthy_capacity", "no healthy worker")
        result = _text(
            self.client.eval(
                RESERVE_SCRIPT,
                1,
                self.state_key,
                request.request_id,
                tenant.tenant_id,
                request.estimated_tokens,
                self.global_queue_capacity,
                tenant.max_queued_requests,
                tenant.max_concurrent_requests,
                tenant.token_quota,
            )
        )
        reasons = {
            "admitted": "request capacity reserved",
            "already_admitted": "reservation already exists",
            "global_queue_full": "global queue is full",
            "tenant_queue_full": "tenant queue is full",
            "tenant_concurrency_limit": "tenant concurrency limit reached",
            "tenant_token_quota": "tenant token quota exceeded",
        }
        if result not in reasons:
            raise RuntimeError(f"unexpected shared admission result: {result}")
        return AdmissionDecision(result in {"admitted", "already_admitted"}, result, reasons[result])

    def mark_running(self, request_id: str) -> None:
        result = _text(
            self.client.eval(
                MARK_RUNNING_SCRIPT,
                1,
                self.state_key,
                request_id,
            )
        )
        if result == "missing":
            raise KeyError(request_id)
        if result == "tenant_concurrency_limit":
            raise RuntimeError("tenant concurrency limit reached")
        if result not in {"running", "already_running"}:
            raise RuntimeError(f"unexpected shared admission result: {result}")

    def release(self, request_id: str) -> None:
        self.client.eval(RELEASE_SCRIPT, 1, self.state_key, request_id)

    def snapshot(self, tenant_id: str) -> dict[str, int]:
        if tenant_id not in self.tenants:
            raise KeyError(tenant_id)
        raw = self.client.eval(SNAPSHOT_SCRIPT, 1, self.state_key, tenant_id)
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise RuntimeError("unexpected shared admission snapshot")
        queued, running, tokens = (int(_text(value)) for value in raw)
        return {"queued": queued, "running": running, "reserved_tokens": tokens}
