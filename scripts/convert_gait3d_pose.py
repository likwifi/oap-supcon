#!/usr/bin/env python3
"""Convert Gait3D's released 2D-pose text archive to canonical OAP NPZ."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oap_supcon.gait3d import convert_gait3d_pose  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--partition", type=Path, default=ROOT / "data/gait3d/Gait3D.json")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--audit-only", action="store_true", help="check partition coverage without parsing poses")
    parser.add_argument("--allow-incomplete", action="store_true", help="export an explicitly labeled available-pose subset")
    args = parser.parse_args()
    report = args.report or args.destination.with_suffix(".coverage.json")
    convert_gait3d_pose(args.source, args.partition, args.destination,
                       allow_incomplete=args.allow_incomplete, report_path=report, audit_only=args.audit_only)


if __name__ == "__main__":
    main()
