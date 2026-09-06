from benchmark.qualification import run_qualification


def test_local_qualification_drills_pass_and_release_all_capacity():
    result = run_qualification(iterations=200)

    assert result.passed
    assert result.kv_final_used_blocks == 0
    assert result.prefix_final_entries == 0
    assert result.admission_final_reserved_tokens == 0
    assert result.injected_failure_observed
    assert result.post_failure_recovery_passed
    assert result.classification.startswith("measured_local_python")
