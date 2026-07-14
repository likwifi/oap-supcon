#!/usr/bin/env python3
"""Render one CASIA-B HRNet frame for manual joint-map inspection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oap_supcon.casia_b import load_pose_pickle  # noqa: E402


COCO_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="one view.pkl from CASIA-B_HRNet")
    parser.add_argument("output", type=Path, help="output PNG path")
    parser.add_argument("--frame", type=int, help="zero-based frame index; default is middle")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required only for this optional renderer") from error

    pose = load_pose_pickle(args.source)
    frame = len(pose) // 2 if args.frame is None else args.frame
    if not 0 <= frame < len(pose):
        raise SystemExit(f"frame must be between 0 and {len(pose) - 1}")
    xy = pose[frame, :, :2]
    confidence = pose[frame, :, 2].clip(0, 1)

    figure, axis = plt.subplots(figsize=(5, 6))
    for start, end in COCO_EDGES:
        axis.plot(xy[[start, end], 0], xy[[start, end], 1], color="0.65", linewidth=2)
    points = axis.scatter(
        xy[:, 0], xy[:, 1], c=confidence, vmin=0, vmax=1, cmap="viridis", s=55
    )
    for index, (horizontal, vertical) in enumerate(xy):
        axis.annotate(str(index), (horizontal, vertical), xytext=(4, 2), textcoords="offset points")
    axis.invert_yaxis()
    axis.set_aspect("equal")
    axis.set_title(f"{args.source.parent.parent.parent.name} frame {frame}")
    figure.colorbar(points, ax=axis, label="clipped HRNet confidence")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
