#!/usr/bin/env python3
"""Validate/copy an already-mapped pose NPZ into the package's canonical schema."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from oap_supcon.data import PoseData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    data = PoseData.load(args.source)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.destination)
    print(f"validated {data.x.shape}; checksum={data.checksum()}")


if __name__ == "__main__":
    main()

