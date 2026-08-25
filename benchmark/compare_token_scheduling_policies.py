#!/usr/bin/env python3
"""Deterministic policy-decision comparison; never presented as hardware performance."""
from __future__ import annotations

import json

from inference.token_scheduling import PrefillDecodePolicy, TokenResourceSnapshot
from inference.token_types import SamplingParameters, SequencePhase, SequenceState


def _sequence(request_id: str, phase: SequencePhase, created_at: float) -> SequenceState:
    sequence = SequenceState(
        request_id,
        "deterministic prompt",
        SamplingParameters(8),
        deadline=10_000,
        created_at=created_at,
    )
    sequence.phase = phase
    return sequence


def build_report() -> dict[str, object]:
    waiting = [
        _sequence(f"waiting-{index}", SequencePhase.QUEUED, index)
        for index in range(3)
    ]
    running = [
        _sequence(f"running-{index}", SequencePhase.DECODE, index)
        for index in range(3)
    ]
    resources = TokenResourceSnapshot(2, 64, 2)
    policies: dict[str, object] = {}
    for name in ("prefill_first", "decode_first", "balanced"):
        policy = PrefillDecodePolicy(name)
        decisions = [policy.select(waiting, running, resources) for _ in range(6)]
        policies[name] = {
            "decisions": [decision.reason for decision in decisions],
            "prefill_slots_selected": sum(
                len(decision.prefill_ids) for decision in decisions
            ),
            "decode_slots_selected": sum(
                len(decision.decode_ids) for decision in decisions
            ),
        }
    return {
        "classification": "deterministic_policy_decisions_not_hardware_performance",
        "workload": {
            "waiting_sequences": 3,
            "running_sequences": 3,
            "iterations": 6,
            "max_batch_size": 2,
            "free_kv_blocks": 64,
        },
        "policies": policies,
    }


def main() -> None:
    print(json.dumps(build_report(), indent=2))


if __name__ == "__main__":
    main()
