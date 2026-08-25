import random

import pytest

from inference.kv_cache import (
    KVBlockAllocator,
    KVCacheCapacityError,
    KVCacheOwnershipError,
)


def test_allocate_release_reuse_and_accounting():
    allocator = KVBlockAllocator(4, 4, bytes_per_block=128)
    first = allocator.allocate("first", 5)
    assert len(first) == 2
    assert allocator.snapshot().bytes_used == 256
    assert allocator.release("first")
    assert not allocator.release("first")

    second = allocator.allocate("second", 16)
    assert set(second) == {0, 1, 2, 3}
    allocator.assert_invariants()


def test_failed_allocation_is_atomic_and_counted():
    allocator = KVBlockAllocator(2, 4)
    allocator.allocate("existing", 4)

    with pytest.raises(KVCacheCapacityError):
        allocator.allocate("too-large", 8)

    assert allocator.block_table("existing") == (0,)
    snapshot = allocator.snapshot()
    assert snapshot.used_blocks == 1
    assert snapshot.allocation_failures == 1
    with pytest.raises(KVCacheOwnershipError):
        allocator.block_table("too-large")


def test_shared_reference_counts_and_watermark_hysteresis():
    allocator = KVBlockAllocator(4, 1, high_watermark=0.75, low_watermark=0.25)
    blocks = allocator.allocate("prefix", 3)
    assert allocator.snapshot().pressure
    allocator.retain("request", blocks[:1])
    allocator.release("prefix")
    assert allocator.snapshot().used_blocks == 1
    assert not allocator.snapshot().pressure
    allocator.release("request")
    allocator.assert_invariants()


def test_shared_table_rejects_duplicate_block_references():
    allocator = KVBlockAllocator(4, 1)
    block = allocator.allocate("prefix", 1)[0]
    with pytest.raises(ValueError, match="unique"):
        allocator.retain("duplicate", (block, block))
    allocator.assert_invariants()


def test_resize_failure_does_not_partially_change_table():
    allocator = KVBlockAllocator(3, 2)
    original = allocator.allocate("request", 2)
    allocator.allocate("other", 4)

    with pytest.raises(KVCacheCapacityError):
        allocator.resize("request", 4)

    assert allocator.block_table("request") == original
    allocator.assert_invariants()


def test_seeded_randomized_ownership_invariants():
    allocator = KVBlockAllocator(32, 4)
    randomizer = random.Random(20260824)
    owners: set[str] = set()
    for _index in range(500):
        owner = f"request-{randomizer.randrange(20)}"
        if owner in owners and randomizer.random() < 0.55:
            allocator.release(owner)
            owners.remove(owner)
        elif owner not in owners:
            try:
                allocator.allocate(owner, randomizer.randrange(1, 17))
                owners.add(owner)
            except KVCacheCapacityError:
                pass
        allocator.assert_invariants()
        assert allocator.snapshot().used_blocks <= allocator.total_blocks

    for owner in owners:
        allocator.release(owner)
    allocator.assert_invariants()
    assert allocator.snapshot().used_blocks == 0
