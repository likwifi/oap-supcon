from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import resolve_dataset_path
from .experiment import load_configuration, run_experiment
from .reporting import aggregate_results, audit_dataset
from .smoke import make_smoke_dataset


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="OAP-SupCon reproducible experiment runner")
    sub = root.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("make-smoke", help="create a tiny synthetic pipeline-test dataset")
    smoke.add_argument("--seed", type=int, default=2026)
    run = sub.add_parser("run", help="train and evaluate one dataset/method/seed")
    run.add_argument("--dataset", required=True, choices=["smoke", "casia_b_pose", "oumvlp_pose", "sustech1k", "gait3d", "grew_pose", "ccpg"])
    run.add_argument("--method", required=True)
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--epochs", type=int)
    run.add_argument("--device", default="auto")
    run.add_argument("--corruption-realizations", type=int)
    audit = sub.add_parser("audit", help="validate standardized data and create an audit report")
    audit.add_argument("--dataset", required=True, choices=["smoke", "casia_b_pose", "oumvlp_pose", "sustech1k", "gait3d", "grew_pose", "ccpg"])
    audit.add_argument("--output")
    sub.add_parser("aggregate", help="regenerate CSV and LaTeX summaries from completed runs")
    return root


def main():
    args = parser().parse_args()
    root = project_root()
    if args.command == "make-smoke":
        path = make_smoke_dataset(root / "data/smoke/processed/dataset.npz", args.seed)
        print(path)
    elif args.command == "run":
        cfg = load_configuration(root, args.dataset, args.method)
        run_dir, metrics = run_experiment(root, cfg, args.seed, args.epochs, args.device, args.corruption_realizations)
        print(json.dumps({"run_dir": str(run_dir), "status": metrics["status"]}, indent=2))
    elif args.command == "audit":
        cfg = load_configuration(root, args.dataset, "ce")
        path = resolve_dataset_path(root, cfg["dataset"])
        output = Path(args.output) if args.output else root / "audits" / args.dataset
        frame = audit_dataset(path, output)
        for row in frame:
            print(row)
    elif args.command == "aggregate":
        frame, _ = aggregate_results(root / "results")
        print(f"aggregated {len(frame)} result rows into {root / 'results/runs.csv'}")


if __name__ == "__main__":
    main()
