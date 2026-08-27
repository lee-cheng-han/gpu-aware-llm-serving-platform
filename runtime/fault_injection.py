from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from runtime.base import RuntimeCapacity, RuntimeResult, RuntimeStreamChunk
from serving_platform.domain import DeviceType, ModelDefinition, RuntimeType


class InjectedRuntimeFault(RuntimeError):
    def __init__(self, operation: str, call_number: int, code: str) -> None:
        super().__init__(f"injected {operation} fault on call {call_number}: {code}")
        self.operation = operation
        self.call_number = call_number
        self.code = code


@dataclass(frozen=True)
class FaultRule:
    operation: str
    fail_on_calls: tuple[int, ...] = ()
    delay_seconds: float = 0
    code: str = "controlled_failure"

    def __post_init__(self) -> None:
        if not self.operation or not self.code:
            raise ValueError("fault operation and code are required")
        if self.delay_seconds < 0 or any(call <= 0 for call in self.fail_on_calls):
            raise ValueError("fault calls must be positive and delay must be non-negative")
        if len(self.fail_on_calls) != len(set(self.fail_on_calls)):
            raise ValueError("fault call numbers must be unique")


class FaultInjectingRuntime:
    """Deterministic runtime decorator used for local recovery and overload drills."""

    def __init__(
        self,
        runtime,
        rules: Sequence[FaultRule],
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if len({rule.operation for rule in rules}) != len(rules):
            raise ValueError("only one fault rule is allowed per operation")
        self.runtime = runtime
        self.runtime_type: RuntimeType = runtime.runtime_type
        self.device_type: DeviceType = runtime.device_type
        self.rules = {rule.operation: rule for rule in rules}
        self._calls: Counter[str] = Counter()
        self._sleep = sleeper

    def _before(self, operation: str) -> None:
        self._calls[operation] += 1
        rule = self.rules.get(operation)
        if rule is None:
            return
        if rule.delay_seconds:
            self._sleep(rule.delay_seconds)
        if self._calls[operation] in rule.fail_on_calls:
            raise InjectedRuntimeFault(operation, self._calls[operation], rule.code)

    def calls(self, operation: str) -> int:
        return self._calls[operation]

    def load_model(self, model: ModelDefinition) -> None:
        self._before("load_model")
        self.runtime.load_model(model)

    def unload_model(self, model_id: str) -> None:
        self._before("unload_model")
        self.runtime.unload_model(model_id)

    def warmup_model(self, model_id: str) -> None:
        self._before("warmup_model")
        self.runtime.warmup_model(model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        self._before("is_model_loaded")
        return bool(self.runtime.is_model_loaded(model_id))

    def count_prompt_tokens(self, model_id: str, prompt: str) -> int:
        self._before("count_prompt_tokens")
        return int(self.runtime.count_prompt_tokens(model_id, prompt))

    def generate(
        self,
        model_id: str,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
    ) -> list[RuntimeResult]:
        self._before("generate")
        return self.runtime.generate(model_id, prompts, max_new_tokens, temperature)

    def stream(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterable[RuntimeStreamChunk]:
        self._before("stream")
        yield from self.runtime.stream(model_id, prompt, max_new_tokens, temperature)

    def capacity(self) -> RuntimeCapacity:
        self._before("capacity")
        return self.runtime.capacity()
