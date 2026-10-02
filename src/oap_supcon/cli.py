from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import resolve_dataset_path
from .experiment import load_configuration, run_experiment
from .reporting import aggregate_results, audit_dataset
from .smoke import make_smoke_dataset
from .model import BACKBONES
from .descriptor_controls import run_descriptor_controls
from .head_training import HEAD_MODES, run_head_training


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
    run.add_argument("--backbone", choices=list(BACKBONES), help="override recipe backbone")
    run.add_argument("--preset", help="recipe name under configs/recipes, e.g. benchmark_v2")
    run.add_argument("--final", action="store_true", help="train on all train identities and evaluate test; requires fixed --epochs")
    audit = sub.add_parser("audit", help="validate standardized data and create an audit report")
    audit.add_argument("--dataset", required=True, choices=["smoke", "casia_b_pose", "oumvlp_pose", "sustech1k", "gait3d", "grew_pose", "ccpg"])
    audit.add_argument("--output")
    aggregate = sub.add_parser("aggregate", help="regenerate CSV and LaTeX summaries from completed runs")
    aggregate.add_argument("--results-root", type=Path, default=Path("results_v2"))
    aggregate.add_argument("--output", type=Path)
    aggregate.add_argument("--duplicates", choices=["first", "last", "error"], default="first")
    controls = sub.add_parser(
        "descriptor-controls", help="run P0 descriptor controls from one saved baseline checkpoint"
    )
    controls.add_argument("--checkpoint", type=Path, required=True)
    controls.add_argument("--output-root", type=Path, default=Path("results_descriptor_controls"))
    controls.add_argument("--device", default="auto")
    controls.add_argument("--corruption-realizations", type=int)
    controls.add_argument("--random-head-seeds", default=",".join(str(v) for v in range(1001, 1011)))
    controls.add_argument("--fusion-weights", default="0,0.25,0.5,0.75,1")
    heads = sub.add_parser("train-part-head", help="run P1 training with a frozen baseline backbone")
    heads.add_argument("--checkpoint", type=Path, required=True)
    heads.add_argument("--mode", choices=HEAD_MODES, required=True)
    heads.add_argument("--output-root", type=Path, default=Path("results_head_controls"))
    heads.add_argument("--device", default="auto")
    heads.add_argument("--epochs", type=int, default=40)
    heads.add_argument("--learning-rate", type=float, default=1e-3)
    heads.add_argument("--variance-floor", type=float, default=1e-4)
    heads.add_argument("--variance-weight", type=float, default=.1)
    heads.add_argument("--corruption-realizations", type=int)
    return root


def main():
    args = parser().parse_args()
    root = project_root()
    if args.command == "make-smoke":
        path = make_smoke_dataset(root / "data/smoke/processed/dataset.npz", args.seed)
        print(path)
    elif args.command == "run":
        cfg = load_configuration(root, args.dataset, args.method, args.preset)
        if args.backbone:
            cfg["model"]["backbone"] = args.backbone
            cfg["model"].pop("options", None)
        if args.final:
            if args.epochs is None:
                raise SystemExit("--final requires --epochs fixed from validation")
            cfg.setdefault("validation", {})["enabled"] = False
            cfg["evaluation"]["split"] = "test"
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
        frame, _ = aggregate_results(root / args.results_root, args.duplicates, args.output)
        print(f"aggregated {len(frame)} result rows into {args.output or root / args.results_root}")
    elif args.command == "descriptor-controls":
        random_head_seeds = [int(value) for value in args.random_head_seeds.split(",")]
        fusion_weights = [float(value) for value in args.fusion_weights.split(",")]
        run_dir, metrics = run_descriptor_controls(
            root, args.checkpoint, args.output_root, args.device,
            random_head_seeds, fusion_weights, args.corruption_realizations,
        )
        print(json.dumps({"run_dir": str(run_dir), "status": metrics["status"]}, indent=2))
    elif args.command == "train-part-head":
        run_dir, metrics = run_head_training(
            root, args.checkpoint, args.output_root, args.mode, args.device,
            args.epochs, args.learning_rate, args.variance_floor,
            args.variance_weight, args.corruption_realizations,
        )
        print(json.dumps({"run_dir": str(run_dir), "status": metrics["status"]}, indent=2))


if __name__ == "__main__":
    main()
