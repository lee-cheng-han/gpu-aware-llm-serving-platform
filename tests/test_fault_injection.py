import pytest

from runtime.fault_injection import (
    FaultInjectingRuntime,
    FaultRule,
    InjectedRuntimeFault,
)
from runtime.simulated_gpu import SimulatedGpuConfig, SimulatedGpuRuntime
from serving_platform.domain import ModelDefinition, RuntimeType


def model():
    return ModelDefinition(
        "model", "main", RuntimeType.SIMULATED, 100, ("float16",), "float16",
        32, 64, True, True, 1, 60,
    )


def test_fault_rule_fails_only_configured_call_and_then_recovers():
    definition = model()
    base = SimulatedGpuRuntime([definition], SimulatedGpuConfig(total_memory_bytes=1000))
    runtime = FaultInjectingRuntime(
        base,
        [FaultRule("generate", fail_on_calls=(2,), code="device_lost")],
    )
    runtime.load_model(definition)
    runtime.generate("model", ["hello"], 1, 0)
    with pytest.raises(InjectedRuntimeFault) as caught:
        runtime.generate("model", ["hello"], 1, 0)
    assert caught.value.code == "device_lost"
    runtime.generate("model", ["hello"], 1, 0)
    assert runtime.calls("generate") == 3


def test_fault_delay_uses_injected_clock_without_sleeping():
    delays = []
    runtime = FaultInjectingRuntime(
        SimulatedGpuRuntime([model()]),
        [FaultRule("capacity", delay_seconds=0.25)],
        sleeper=delays.append,
    )
    runtime.capacity()
    assert delays == [0.25]
