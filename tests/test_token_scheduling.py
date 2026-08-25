from inference.token_scheduling import PrefillDecodePolicy, TokenResourceSnapshot
from inference.token_types import SamplingParameters, SequencePhase, SequenceState


def sequence(request_id, phase):
    result = SequenceState(request_id, "prompt", SamplingParameters(2), 100)
    result.phase = phase
    return result


RESOURCES = TokenResourceSnapshot(2, 10, 2)


def test_decode_first_and_prefill_first_are_explicit():
    waiting = [sequence("waiting", SequencePhase.QUEUED)]
    running = [sequence("running", SequencePhase.DECODE)]

    decode = PrefillDecodePolicy("decode_first").select(waiting, running, RESOURCES)
    prefill = PrefillDecodePolicy("prefill_first").select(waiting, running, RESOURCES)

    assert decode.decode_ids == ("running",)
    assert prefill.prefill_ids == ("waiting",)


def test_balanced_policy_alternates_workload_classes():
    policy = PrefillDecodePolicy("balanced")
    waiting = [sequence("waiting", SequencePhase.QUEUED)]
    running = [sequence("running", SequencePhase.DECODE)]

    first = policy.select(waiting, running, RESOURCES)
    second = policy.select(waiting, running, RESOURCES)

    assert first.decode_ids == ("running",)
    assert second.prefill_ids == ("waiting",)


def test_policy_respects_iteration_batch_limit():
    waiting = [sequence(str(index), SequencePhase.QUEUED) for index in range(4)]
    decision = PrefillDecodePolicy("prefill_first").select(
        waiting, [], TokenResourceSnapshot(4, 20, 2)
    )
    assert decision.prefill_ids == ("0", "1")
