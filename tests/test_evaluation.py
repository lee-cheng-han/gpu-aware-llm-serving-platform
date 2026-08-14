import json
from pathlib import Path

from benchmark.evaluation import check_gates, load_workload, simulate, write_report

WORKLOAD = Path("benchmark/workloads/heterogeneous.json")


def test_simulated_evaluation_is_deterministic_and_passes_gates():
    document = load_workload(WORKLOAD)
    first = simulate(document)
    second = simulate(document)
    first["metrics"].pop("scheduler_decision_p95_us")
    second["metrics"].pop("scheduler_decision_p95_us")
    assert first == second
    result = simulate(document)
    assert check_gates(result) == []
    assert result["classification"] == "deterministic_simulation_not_real_hardware"


def test_report_outputs_are_labelled_and_machine_readable(tmp_path):
    result = simulate(load_workload(WORKLOAD))
    assert write_report(result, tmp_path) == []
    assert json.loads((tmp_path / "result.json").read_text())["classification"].startswith(
        "deterministic_simulation"
    )
    assert "not a real GPU benchmark" in (tmp_path / "report.md").read_text()
    assert "<svg" in (tmp_path / "tenant_fairness.svg").read_text()
