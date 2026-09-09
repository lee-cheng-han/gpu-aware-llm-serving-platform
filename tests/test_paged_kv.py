import pytest

from inference.paged_kv import PagedKVConfig, TorchPagedKVCache


def test_paged_kv_configuration_rejects_invalid_layouts():
    with pytest.raises(ValueError, match="dimensions"):
        PagedKVConfig(0, 4, 2, 2, 8)
    with pytest.raises(ValueError, match="device"):
        PagedKVConfig(4, 4, 2, 2, 8, device="remote")
    with pytest.raises(ValueError, match="dtype"):
        PagedKVConfig(4, 4, 2, 2, 8, dtype="float64")


def test_physical_pages_gather_different_length_block_tables():
    torch = pytest.importorskip("torch")
    cache = TorchPagedKVCache(PagedKVConfig(8, 2, 2, 2, 4))
    cache.allocate("short", 2)
    cache.allocate("long", 5)
    short_keys = tuple(torch.full((2, 2, 4), layer + 1.0) for layer in range(2))
    short_values = tuple(torch.full((2, 2, 4), layer + 3.0) for layer in range(2))
    long_keys = tuple(torch.full((2, 5, 4), layer + 5.0) for layer in range(2))
    long_values = tuple(torch.full((2, 5, 4), layer + 7.0) for layer in range(2))

    cache.append("short", short_keys, short_values)
    cache.append("long", long_keys, long_values)
    keys, values, mask = cache.materialize(1, ("short", "long"))

    assert keys.shape == values.shape == (2, 2, 5, 4)
    assert mask[:, 0, 0, :].tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, True],
    ]
    assert torch.equal(keys[0, :, :2, :], short_keys[1])
    assert torch.equal(values[1], long_values[1])
    cache.assert_invariants()
    assert cache.snapshot().stored_tokens == 7
    assert cache.release("short")
    assert cache.release("long")
    assert cache.snapshot().used_blocks == 0


def test_reference_paged_attention_matches_materialized_sdpa():
    torch = pytest.importorskip("torch")
    cache = TorchPagedKVCache(PagedKVConfig(4, 2, 1, 2, 4))
    cache.allocate("sequence", 3)
    keys = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 24,)
    values = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 12,)
    cache.append("sequence", keys, values)
    query = torch.ones((1, 2, 1, 4), dtype=torch.float32)

    actual = cache.attention(0, ("sequence",), query)
    gathered_keys, gathered_values, mask = cache.materialize(0, ("sequence",))
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        gathered_keys,
        gathered_values,
        attn_mask=mask,
        dropout_p=0,
        is_causal=False,
    )

    assert torch.equal(actual, expected)
    assert actual.shape == (1, 2, 1, 4)


def test_paged_kv_rejects_capacity_overrun_before_writing():
    torch = pytest.importorskip("torch")
    cache = TorchPagedKVCache(PagedKVConfig(2, 2, 1, 1, 2))
    cache.allocate("sequence", 2)
    tensor = torch.zeros((1, 3, 2))

    with pytest.raises(ValueError, match="reserved"):
        cache.append("sequence", (tensor,), (tensor,))

    assert cache.sequence("sequence").tokens == 0
