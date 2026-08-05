#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from oap_supcon.experiment import load_configuration, run_experiment


SEARCH = [
    ("part_weight", 0.25), ("part_weight", 0.5), ("part_weight", 1.0),
    ("temporal_weight", 0.1), ("temporal_weight", 0.2), ("temporal_weight", 0.5),
    ("ce_weight", 0.5), ("ce_weight", 1.0), ("ce_weight", 2.0),
    ("temperature", 0.05), ("temperature", 0.1), ("temperature", 0.2),
    ("embedding_dim", 128), ("embedding_dim", 256), ("embedding_dim", 512),
    ("train_severity", 0.2), ("train_severity", 0.5), ("train_severity", 0.7),
]


def main():
    parser = argparse.ArgumentParser(description="One-factor sensitivity around the OAP-SupCon base configuration")
    parser.add_argument("--dataset", default="casia_b_pose")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    parameter, value = SEARCH[args.task_id]
    root = Path(__file__).resolve().parents[1]
    cfg = load_configuration(root, args.dataset, "oap_supcon")
    if parameter in cfg["method"]:
        cfg["method"][parameter] = value
    elif parameter in cfg["train"]:
        cfg["train"][parameter] = value
    else:
        cfg["model"][parameter] = value
    cfg["method_name"] = f"oap_sensitivity_{parameter}_{value}"
    run_dir, _ = run_experiment(root, cfg, args.seed, None, args.device, None)
    print(run_dir)


if __name__ == "__main__":
    main()

