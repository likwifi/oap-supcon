#!/usr/bin/env python3
"""Export a project data split and configs for official FastPoseGait baselines.

CASIA-B `final` reproduces published scores on the official 74/50 protocol.
Gait3D `final` scores the available-pose test subset used by this project.
`validation` exports only the fixed development holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oap_supcon.reference_baseline import prepare_fastposegait


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new directory for poses/configs/manifest")
    parser.add_argument("--dataset", choices=["casia_b_pose", "gait3d"], default="casia_b_pose")
    parser.add_argument("--stage", choices=["validation", "final"], default="validation")
    parser.add_argument("--models", help="comma-separated; defaults: CASIA-B gpgait,gpgait++; Gait3D gpgait,gaitgraph2")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    models = tuple(args.models.split(",")) if args.models else None
    manifest = prepare_fastposegait(ROOT, args.output, models, args.gpus, args.dry_run,
                                    dataset=args.dataset, stage=args.stage)
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in ("training_identities", "evaluated_identities", "validation_identities")},
                     indent=2))


if __name__ == "__main__":
    main()
