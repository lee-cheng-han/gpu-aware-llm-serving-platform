from inference.prefix_cache import PrefixCache


def test_exact_lookup_tracks_hits_and_misses():
    cache = PrefixCache(max_entries=2, max_tokens=8)
    assert cache.get((1, 2)) is None
    assert cache.put((1, 2), "state")
    assert cache.get((1, 2)) == "state"

    snapshot = cache.snapshot()
    assert snapshot.entries == 1
    assert snapshot.cached_tokens == 2
    assert snapshot.hits == snapshot.misses == 1
    assert snapshot.hit_rate == 0.5


def test_lru_eviction_respects_entry_and_token_bounds():
    cache = PrefixCache(max_entries=2, max_tokens=4)
    cache.put((1, 2), "first")
    cache.put((3,), "second")
    assert cache.get((1, 2)) == "first"
    cache.put((4, 5), "third")

    assert cache.get((3,)) is None
    assert cache.get((1, 2)) == "first"
    assert cache.get((4, 5)) == "third"
    snapshot = cache.snapshot()
    assert snapshot.entries == 2
    assert snapshot.cached_tokens == 4
    assert snapshot.evictions == 1


def test_radix_lookup_returns_the_longest_cached_token_prefix():
    cache = PrefixCache(max_entries=4, max_tokens=32)
    cache.put((1, 2), "short")
    cache.put((1, 2, 3, 4), "long")

    match = cache.longest_prefix((1, 2, 3, 4, 5, 6))

    assert match is not None
    assert match.backend_state == "long"
    assert match.token_ids == (1, 2, 3, 4)
    assert match.matched_tokens == 4
    assert not match.exact
    snapshot = cache.snapshot()
    assert snapshot.partial_hits == 1
    assert snapshot.matched_tokens == 4


def test_disabled_and_oversized_entries_are_bypassed():
    disabled = PrefixCache()
    assert not disabled.put((1,), object())
    assert disabled.get((1,)) is None
    assert disabled.snapshot().bypasses == 2

    bounded = PrefixCache(max_entries=1, max_tokens=2)
    assert not bounded.put((1, 2, 3), object())
    assert bounded.snapshot().entries == 0
