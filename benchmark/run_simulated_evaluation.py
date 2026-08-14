#!/usr/bin/env python3
"""Run a labelled deterministic simulation and optionally enforce its regression gates."""

import argparse
from pathlib import Path

from benchmark.evaluation import load_workload, simulate, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload", type=Path, default=Path("benchmark/workloads/heterogeneous.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark/results/simulated/heterogeneous")
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = simulate(load_workload(args.workload))
    failures = write_report(result, args.output_dir)
    print(args.output_dir / "report.md")
    if args.check and failures:
        parser.error("regression gates failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
