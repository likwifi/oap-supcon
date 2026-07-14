#!/usr/bin/env python3
"""Convert the FastPoseGait/ScienceDB CASIA-B HRNet release to OAP NPZ."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oap_supcon.casia_b import convert_casia_b_hrnet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="extracted CASIA-B_HRNet directory containing 001/... through 124/...",
    )
    parser.add_argument("destination", type=Path, help="output dataset.npz path")
    parser.add_argument(
        "--no-strict-release-check",
        action="store_true",
        help="allow missing sequences beyond the three known gaps in the official release",
    )
    args = parser.parse_args()
    data = convert_casia_b_hrnet(
        args.source,
        args.destination,
        strict_release=not args.no_strict_release_check,
    )
    counts = {
        name: int((data.split == name).sum()) for name in ("train", "gallery", "probe")
    }
    print(f"validated canonical shape={data.x.shape}; splits={counts}")
    print(f"canonical checksum={data.checksum()}")


if __name__ == "__main__":
    main()
