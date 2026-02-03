#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def load_results(path: str, mode: str) -> dict[tuple[str, int], float]:
    with open(path, "r") as f:
        data = json.load(f)
    results = data["results"][mode]
    return {(r["name"], r["batch_size"]): r["mean_ms"] for r in results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--baseline-mode", default="no_compile")
    parser.add_argument("--compare-mode", default="no_compile")
    parser.add_argument("--batch-sizes", required=True)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    ops = [
        ("forward_vertices", "fwd_v"),
        ("forward_skeleton", "fwd_s"),
        ("backward_vertices", "bwd_v"),
        ("backward_skeleton", "bwd_s"),
    ]

    baseline = load_results(args.baseline, args.baseline_mode)
    target = load_results(args.target, args.compare_mode)

    print(f"Summary (speedup = baseline / target, mode={args.compare_mode})")
    for bs in batch_sizes:
        parts = []
        for op, short in ops:
            base = baseline.get((op, bs))
            cur = target.get((op, bs))
            if base is None or cur is None:
                parts.append(f"{short} n/a")
            else:
                parts.append(f"{short} {base / cur:.2f}x")
        print(f"B={bs}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
