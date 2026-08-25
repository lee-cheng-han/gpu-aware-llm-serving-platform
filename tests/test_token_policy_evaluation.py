from benchmark.compare_token_scheduling_policies import build_report


def test_token_policy_report_is_deterministic_and_distinguishes_tradeoffs():
    first = build_report()
    assert first == build_report()
    policies = first["policies"]
    assert policies["prefill_first"]["decode_slots_selected"] == 0
    assert policies["decode_first"]["prefill_slots_selected"] == 0
    assert policies["balanced"]["prefill_slots_selected"] == 6
    assert policies["balanced"]["decode_slots_selected"] == 6
