from types import SimpleNamespace

import pytest

from runtime.distributed import (
    DistributedBackend,
    DistributedConfig,
    DistributedContext,
    balanced_shard,
)


class FakeDistributed:
    class ReduceOp:
        SUM = "sum"

    def __init__(self, *, nccl=True):
        self.nccl = nccl
        self.initialized = False
        self.init_arguments = None
        self.destroyed = False
        self.reductions = []
        self.broadcasts = []
        self.gathered = []
        self.barriers = 0

    def is_available(self):
        return True

    def is_nccl_available(self):
        return self.nccl

    def is_initialized(self):
        return self.initialized

    def init_process_group(self, **arguments):
        self.initialized = True
        self.init_arguments = arguments

    def get_world_size(self):
        return self.init_arguments["world_size"]

    def get_rank(self):
        return self.init_arguments["rank"]

    def get_backend(self):
        return self.init_arguments["backend"]

    def all_reduce(self, tensor, op):
        self.reductions.append((tensor, op))

    def broadcast(self, tensor, src):
        self.broadcasts.append((tensor, src))

    def barrier(self):
        self.barriers += 1

    def all_gather_object(self, output, value):
        output[:] = [value, value]
        self.gathered.append(value)

    def destroy_process_group(self):
        self.destroyed = True
        self.initialized = False


class FakeCuda:
    def __init__(self, available=True, devices=2):
        self.available = available
        self.devices = devices
        self.selected = None

    def is_available(self):
        return self.available

    def device_count(self):
        return self.devices

    def set_device(self, index):
        self.selected = index


def fake_torch(distributed=None, cuda=None):
    return SimpleNamespace(
        distributed=distributed or FakeDistributed(),
        cuda=cuda or FakeCuda(),
    )


def test_balanced_tensor_shards_cover_uneven_dimension_without_overlap():
    shards = [balanced_shard(10, rank, 3) for rank in range(3)]

    assert [(shard.start, shard.end) for shard in shards] == [(0, 4), (4, 7), (7, 10)]
    assert sum(shard.size for shard in shards) == 10


def test_single_rank_context_is_dependency_free_identity():
    context = DistributedContext(DistributedConfig(world_size=1))
    status = context.initialize()
    value = object()

    assert status.initialized and not status.enabled
    assert context.all_reduce_sum(value) is value
    assert context.broadcast(value) is value
    context.barrier()
    context.shutdown()
    assert not context.status().initialized


def test_nccl_context_initializes_collectives_and_owned_shutdown():
    distributed = FakeDistributed()
    cuda = FakeCuda()
    torch = fake_torch(distributed, cuda)
    config = DistributedConfig(world_size=2, rank=1, local_rank=1)
    context = DistributedContext(config, module_loader=lambda _name: torch)

    status = context.initialize()
    tensor = object()
    assert status.owns_process_group
    assert cuda.selected == 1
    assert distributed.init_arguments["backend"] == "nccl"
    assert distributed.init_arguments["rank"] == 1
    assert context.all_reduce_sum(tensor) is tensor
    assert context.broadcast(tensor, source_rank=0) is tensor
    assert context.all_gather_objects("device") == ("device", "device")
    context.barrier()
    assert distributed.reductions == [(tensor, "sum")]
    assert distributed.broadcasts == [(tensor, 0)]
    assert distributed.gathered == ["device"]
    assert distributed.barriers == 1

    context.shutdown()
    assert distributed.destroyed


def test_nccl_configuration_never_falls_back_to_cpu():
    torch = fake_torch(cuda=FakeCuda(available=False))
    context = DistributedContext(
        DistributedConfig(world_size=2), module_loader=lambda _name: torch
    )

    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        context.initialize()


def test_distributed_configuration_validates_rank_and_environment(monkeypatch):
    with pytest.raises(ValueError, match="rank"):
        DistributedConfig(world_size=2, rank=2)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("DISTRIBUTED_BACKEND", "gloo")

    config = DistributedConfig.from_environment()

    assert config.world_size == 4
    assert config.rank == 2
    assert config.local_rank == 1
    assert config.backend is DistributedBackend.GLOO
