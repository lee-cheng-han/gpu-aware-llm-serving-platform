from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.qualification import run_qualification


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded local qualification drills")
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--max-python-heap-growth-bytes", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = run_qualification(
        args.iterations,
        args.max_python_heap_growth_bytes,
    )
    rendered = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if args.check and not result.passed:
        parser.error("local qualification gates failed")


if __name__ == "__main__":
    main()
