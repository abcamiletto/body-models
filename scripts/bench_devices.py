#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["csv", "lines"], default="csv")
    args = parser.parse_args()

    devices = ["cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    if args.format == "lines":
        print("\n".join(devices))
    else:
        print(",".join(devices))


if __name__ == "__main__":
    main()
