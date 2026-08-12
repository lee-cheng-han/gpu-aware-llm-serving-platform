import pytest

from serving_platform.domain import ModelDefinition, RuntimeType
from serving_platform.registry import InMemoryModelRegistry


def model(revision: str = "main") -> ModelDefinition:
    return ModelDefinition(
        "model", revision, RuntimeType.SIMULATED, 200, ("float16",), "float16",
        100, 200, True, True, 5, 60,
    )


def test_model_registry_is_sorted_and_idempotent():
    other = ModelDefinition(
        "another", "main", RuntimeType.SIMULATED, 100, ("float16",), "float16",
        100, 200, True, True, 5, 60,
    )
    registry = InMemoryModelRegistry([model(), other])
    registry.register(model())
    assert [definition.model_id for definition in registry.list()] == ["another", "model"]


def test_model_registry_rejects_conflicting_definition():
    registry = InMemoryModelRegistry([model()])
    with pytest.raises(ValueError, match="conflicting"):
        registry.register(model("different"))
