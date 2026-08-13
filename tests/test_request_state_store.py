import pytest

from serving_platform.domain import RequestRecord, RequestState
from serving_platform.request_state import InMemoryRequestStateStore


def request() -> RequestRecord:
    return RequestRecord("request", "tenant", "model", "hello", 1, 1, 0, 100, False)


def test_request_store_uses_isolated_snapshots():
    store = InMemoryRequestStateStore()
    item = request()
    store.create(item)
    item.transition(RequestState.VALIDATED)
    assert store.get(item.request_id).status == RequestState.RECEIVED
    store.save(item)
    assert store.get(item.request_id).status == RequestState.VALIDATED


def test_request_store_rejects_duplicates_and_missing_updates():
    store = InMemoryRequestStateStore()
    item = request()
    store.create(item)
    with pytest.raises(ValueError, match="already exists"):
        store.create(item)
    missing = request()
    missing.request_id = "missing"
    with pytest.raises(KeyError, match="does not exist"):
        store.save(missing)
