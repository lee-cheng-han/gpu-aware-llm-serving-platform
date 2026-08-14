import json

import pytest

from serving_platform.domain import RequestRecord, RequestState
from serving_platform.request_state import RedisRequestStateStore


class FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, *, nx=False, xx=False):
        if nx and key in self.data:
            return False
        if xx and key not in self.data:
            return False
        self.data[key] = value
        return True

    def scan_iter(self, match):
        prefix = match.removesuffix("*")
        return iter(key for key in self.data if key.startswith(prefix))


def request() -> RequestRecord:
    item = RequestRecord(
        "request", "tenant", "model", "top secret prompt", 3, 2, 5, 100, False
    )
    item.transition(RequestState.VALIDATED, 2)
    item.transition(RequestState.ADMITTED, 3)
    item.assigned_worker_id = "worker"
    item.attempt_count = 1
    item.retry_reasons.append("test_retry")
    return item


def test_redis_store_survives_adapter_restart_without_persisting_prompt():
    redis = FakeRedis()
    first_process = RedisRequestStateStore(redis)
    first_process.create(request())
    raw = next(iter(redis.data.values()))
    assert "top secret prompt" not in raw
    assert "generated" not in raw

    restarted_process = RedisRequestStateStore(redis)
    restored = restarted_process.get("request")
    assert restored.prompt == "[redacted]"
    assert restored.payload_available is False
    assert restored.status == RequestState.ADMITTED
    assert restored.assigned_worker_id == "worker"
    assert restored.retry_reasons == ["test_retry"]
    assert json.loads(raw)["transition_timestamps"]["validated"] == 2


def test_redis_store_enforces_create_and_update_existence():
    redis = FakeRedis()
    store = RedisRequestStateStore(redis)
    item = request()
    store.create(item)
    with pytest.raises(ValueError, match="already exists"):
        store.create(item)
    missing = request()
    missing.request_id = "missing"
    with pytest.raises(KeyError, match="does not exist"):
        store.save(missing)
