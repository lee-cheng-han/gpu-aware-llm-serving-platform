import json
from concurrent.futures import ThreadPoolExecutor
from threading import RLock

import pytest

from serving_platform.admission import RedisAdmissionController
from serving_platform.coordination import RedisLease
from serving_platform.domain import (
    DeviceType,
    HealthStatus,
    ModelDefinition,
    RequestRecord,
    RuntimeType,
    TenantLimits,
    WorkerState,
)


class FakeAtomicRedis:
    """Small command-level fake; every script executes under the same server lock."""

    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.values: dict[str, tuple[str, float]] = {}
        self.now_ms = 0.0
        self.lock = RLock()

    def _live_value(self, key: str) -> str | None:
        value = self.values.get(key)
        if value is None:
            return None
        token, expires_at = value
        if expires_at <= self.now_ms:
            self.values.pop(key, None)
            return None
        return token

    def get(self, key: str):
        with self.lock:
            return self._live_value(key)

    def set(self, key: str, value: str, *, nx: bool = False, px: int | None = None):
        with self.lock:
            if nx and self._live_value(key) is not None:
                return False
            self.values[key] = (value, self.now_ms + (px or 0))
            return True

    def eval(self, script: str, numkeys: int, *items: object):
        assert numkeys == 1
        key = str(items[0])
        arguments = [str(item) for item in items[1:]]
        with self.lock:
            if "admission_configure_v1" in script:
                state = self.hashes.setdefault(key, {})
                current = state.get("config:fingerprint")
                if current is None:
                    state["config:fingerprint"] = arguments[0]
                    return b"configured"
                return b"existing" if current == arguments[0] else b"conflict"
            if "admission_reserve_v1" in script:
                return self._reserve(key, arguments)
            if "admission_mark_running_v1" in script:
                return self._mark_running(key, arguments)
            if "admission_release_v1" in script:
                return self._release(key, arguments)
            if "admission_snapshot_v1" in script:
                return self._snapshot(key, arguments)
            if "coordination_renew_v1" in script:
                if self._live_value(key) != arguments[0]:
                    return 0
                self.values[key] = (arguments[0], self.now_ms + int(arguments[1]))
                return 1
            if "coordination_release_v1" in script:
                if self._live_value(key) != arguments[0]:
                    return 0
                self.values.pop(key)
                return 1
        raise AssertionError("unknown script")

    def _reserve(self, key: str, arguments: list[str]):
        state = self.hashes.setdefault(key, {})
        request_id, tenant = arguments[:2]
        tokens, global_limit, queue_limit, concurrency_limit, token_limit = map(
            int, arguments[2:]
        )
        reservation_key = f"reservation:{request_id}"
        queued_key = f"tenant:{tenant}:queued"
        running_key = f"tenant:{tenant}:running"
        tokens_key = f"tenant:{tenant}:tokens"
        if reservation_key in state:
            return b"already_admitted"
        if int(state.get("global:queued", 0)) >= global_limit:
            return b"global_queue_full"
        if int(state.get(queued_key, 0)) >= queue_limit:
            return b"tenant_queue_full"
        if int(state.get(running_key, 0)) >= concurrency_limit:
            return b"tenant_concurrency_limit"
        if int(state.get(tokens_key, 0)) + tokens > token_limit:
            return b"tenant_token_quota"
        self._increment(state, "global:queued", 1)
        self._increment(state, queued_key, 1)
        self._increment(state, tokens_key, tokens)
        state[reservation_key] = json.dumps(
            {
                "tenant": tenant,
                "tokens": tokens,
                "running": False,
                "concurrency_limit": concurrency_limit,
            }
        )
        return b"admitted"

    def _mark_running(self, key: str, arguments: list[str]):
        state = self.hashes.setdefault(key, {})
        reservation_key = f"reservation:{arguments[0]}"
        encoded = state.get(reservation_key)
        if encoded is None:
            return b"missing"
        reservation = json.loads(encoded)
        if reservation["running"]:
            return b"already_running"
        tenant = reservation["tenant"]
        running_key = f"tenant:{tenant}:running"
        if int(state.get(running_key, 0)) >= reservation["concurrency_limit"]:
            return b"tenant_concurrency_limit"
        self._increment(state, "global:queued", -1)
        self._increment(state, f"tenant:{tenant}:queued", -1)
        self._increment(state, running_key, 1)
        reservation["running"] = True
        state[reservation_key] = json.dumps(reservation)
        return b"running"

    def _release(self, key: str, arguments: list[str]):
        state = self.hashes.setdefault(key, {})
        reservation_key = f"reservation:{arguments[0]}"
        encoded = state.get(reservation_key)
        if encoded is None:
            return 0
        reservation = json.loads(encoded)
        tenant = reservation["tenant"]
        if reservation["running"]:
            self._increment(state, f"tenant:{tenant}:running", -1)
        else:
            self._increment(state, "global:queued", -1)
            self._increment(state, f"tenant:{tenant}:queued", -1)
        self._increment(state, f"tenant:{tenant}:tokens", -reservation["tokens"])
        state.pop(reservation_key)
        return 1

    def _snapshot(self, key: str, arguments: list[str]):
        state = self.hashes.setdefault(key, {})
        tenant = arguments[0]
        return [
            state.get(f"tenant:{tenant}:queued", "0"),
            state.get(f"tenant:{tenant}:running", "0"),
            state.get(f"tenant:{tenant}:tokens", "0"),
        ]

    @staticmethod
    def _increment(state: dict[str, str], key: str, amount: int) -> None:
        state[key] = str(int(state.get(key, 0)) + amount)


def model() -> ModelDefinition:
    return ModelDefinition(
        "model", "main", RuntimeType.SIMULATED, 100, ("float16",), "float16",
        100, 200, True, True, 5, 60,
    )


def worker() -> WorkerState:
    return WorkerState(
        "worker", DeviceType.SIMULATED_GPU, "simulated", 1000, 900,
        health_status=HealthStatus.HEALTHY,
    )


def request(request_id: str) -> RequestRecord:
    return RequestRecord(request_id, "tenant", "model", "hello", 9, 1, 0, 100, False)


def test_redis_admission_is_atomic_across_control_plane_instances():
    redis = FakeAtomicRedis()
    limits = [TenantLimits("tenant", 1, 1, 10)]
    controllers = [
        RedisAdmissionController(redis, limits, 1, namespace="integration", clock=lambda: 0)
        for _ in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = list(
            pool.map(
                lambda pair: pair[0].decide(pair[1], model(), [worker()]),
                zip(controllers, (request("one"), request("two")), strict=True),
            )
        )

    assert sum(decision.admitted for decision in decisions) == 1
    assert {decision.code for decision in decisions} == {"admitted", "global_queue_full"}
    admitted_id = "one" if decisions[0].admitted else "two"
    assert controllers[1].snapshot("tenant") == {
        "queued": 1,
        "running": 0,
        "reserved_tokens": 10,
    }

    controllers[1].mark_running(admitted_id)
    controllers[0].mark_running(admitted_id)
    assert controllers[0].snapshot("tenant")["running"] == 1
    controllers[0].release(admitted_id)
    controllers[1].release(admitted_id)
    assert controllers[1].snapshot("tenant") == {
        "queued": 0,
        "running": 0,
        "reserved_tokens": 0,
    }


def test_redis_lease_enforces_ownership_renewal_and_expiry():
    redis = FakeAtomicRedis()
    first = RedisLease(redis, "recovery-supervisor", 1, owner_token="first")
    second = RedisLease(redis, "recovery-supervisor", 1, owner_token="second")

    assert first.acquire()
    assert first.is_owner()
    assert not second.acquire()
    assert not second.renew()
    assert first.renew()
    redis.now_ms += 1001
    assert not first.is_owner()
    assert second.acquire()
    assert not first.release()
    assert second.release()


def test_redis_admission_rejects_configuration_drift_in_one_namespace():
    redis = FakeAtomicRedis()
    RedisAdmissionController(
        redis,
        [TenantLimits("tenant", 1, 1, 10)],
        1,
        namespace="shared",
    )

    with pytest.raises(ValueError, match="configuration conflicts"):
        RedisAdmissionController(
            redis,
            [TenantLimits("tenant", 2, 1, 10)],
            1,
            namespace="shared",
        )
