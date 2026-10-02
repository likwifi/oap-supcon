#!/usr/bin/env python3
"""Select one baseline checkpoint and one P1 head objective for a SLURM task."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


MODES = ("distill", "triplet_variance")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results_v2_factorial"))
    parser.add_argument("--output-root", type=Path, default=Path("results_head_controls"))
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--corruption-realizations", type=int)
    args = parser.parse_args()

    requested = [int(value) for value in args.seeds.split(",")]
    modes = args.modes.split(",")
    unknown = set(modes).difference(MODES)
    if unknown:
        raise SystemExit(f"unknown modes: {sorted(unknown)}")
    checkpoints = {}
    for checkpoint in sorted(args.results_root.glob("*/checkpoint.pt")):
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        if config.get("method_name") == "metric_baseline" and int(config["seed"]) in requested:
            seed = int(config["seed"])
            if seed in checkpoints:
                raise SystemExit(f"duplicate metric_baseline checkpoint for seed {seed}")
            checkpoints[seed] = checkpoint
    missing = sorted(set(requested).difference(checkpoints))
    if missing:
        raise SystemExit(f"missing metric_baseline checkpoints for seeds {missing}")
    combinations = [(mode, seed, checkpoints[seed]) for mode in modes for seed in requested]
    if not 0 <= args.task_id < len(combinations):
        raise SystemExit(f"task {args.task_id} is outside 0..{len(combinations) - 1}")
    mode, _, checkpoint = combinations[args.task_id]
    command = [
        sys.executable, "-m", "oap_supcon.cli", "train-part-head",
        "--checkpoint", str(checkpoint), "--mode", mode,
        "--output-root", str(args.output_root), "--device", args.device,
        "--epochs", str(args.epochs),
    ]
    if args.corruption_realizations is not None:
        command.extend(["--corruption-realizations", str(args.corruption_realizations)])
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
