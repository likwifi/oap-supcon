#!/usr/bin/env python3
"""Validation-only, one-change-at-a-time architecture screen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

VARIANTS = ("control", "angles", "late_fusion", "adaptive", "parts", "pooling", "full")


def architecture_tasks(seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct training seeds")
    return [(variant, seed) for variant in VARIANTS for seed in seeds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="casia_b_pose")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--epochs", type=int, default=80, help="fixed screen budget for every arm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    try:
        tasks = architecture_tasks([int(value) for value in args.seeds.split(",")])
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.task_id < len(tasks):
        parser.error(f"task-id must be between 0 and {len(tasks) - 1}")
    variant, seed = tasks[args.task_id]
    preset = "benchmark_v3" if variant == "full" else f"benchmark_v3_{variant}"
    root = Path(__file__).resolve().parents[1]
    # Dry runs also work before installing the package.
    sys.path.insert(0, str(root / "src"))
    from oap_supcon.experiment import load_configuration
    cfg = load_configuration(root, args.dataset, "metric_baseline", preset)
    if not cfg["validation"].get("enabled") or cfg["evaluation"].get("split") != "validation":
        raise ValueError("the architecture screen requires validation-only recipes")
    command = [sys.executable, "-m", "oap_supcon.cli", "run", "--dataset", args.dataset,
               "--method", "metric_baseline", "--preset", preset, "--seed", str(seed),
               "--epochs", str(args.epochs), "--device", args.device]
    print(json.dumps({"variant": variant, "seed": seed, "task_count": len(tasks),
                      "epochs": args.epochs, "stage": "validation", "command": command}, indent=2), flush=True)
    if not args.dry_run:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(command, check=True, cwd=root, env=env)


if __name__ == "__main__":
    main()
