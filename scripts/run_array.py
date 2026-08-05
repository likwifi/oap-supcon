#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", required=True, help="comma-separated")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args()
    combinations = [(method, int(seed)) for method in args.methods.split(",") for seed in args.seeds.split(",")]
    if args.task_id >= len(combinations):
        raise SystemExit(f"task {args.task_id} is outside 0..{len(combinations) - 1}")
    method, seed = combinations[args.task_id]
    command = [sys.executable, "-m", "oap_supcon.cli", "run", "--dataset", args.dataset, "--method", method, "--seed", str(seed), "--device", "cuda"]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

