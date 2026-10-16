#!/usr/bin/env python3
"""Fix a final-training epoch budget from matched validation curves.

Predeclared rule, applied to the across-seed mean of the checkpoint-selection
score (or `--metric`), smoothed by a centered moving average:

1. If the smoothed curve peaks at the last evaluated epoch and its late slope is
   positive, the curve has not plateaued: extend the validated schedule instead
   of freezing a budget.
2. Otherwise, if the smoothed value at the last epoch is within `--tolerance` of
   the smoothed maximum, freeze the full schedule length. The validated cosine
   schedule only anneals completely at its last epoch, and a peak inside it is
   not the optimum of a shorter schedule.
3. Otherwise, freeze the smoothed argmax epoch, and report that the shorter run
   anneals its learning rate earlier than the validated runs did.

Per-seed best epochs and their median (the earlier convention) are reported
for transparency, but do not set the budget. Curves come from `history.json`,
or from Slurm logs (`--logs`) when only the selection score was printed.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

LOG_LINE = re.compile(r"epoch (\d+)/(\d+):.*? val=([0-9.]+)")
LOG_SEED = re.compile(r"--seed (\d+)")


def curve_from_history(directory: Path, metric: str):
    history = json.loads((directory / "history.json").read_text())
    config = (directory / "config.yaml").read_text()
    seed = int(re.search(r"^seed: (\d+)$", config, re.MULTILINE).group(1))
    points = {int(row["epoch"]): float(row["validation"][metric]) for row in history if "validation" in row}
    return seed, points, int(history[-1]["epoch"])


def curve_from_log(path: Path, metric: str):
    if metric != "score":
        raise ValueError("Slurm logs print only the selection score; use history.json for other metrics")
    text = path.read_text()
    matches = LOG_LINE.findall(text)
    seed = LOG_SEED.search(text)
    if not matches or seed is None:
        raise ValueError(f"no validation curve or seed in {path}")
    total = {int(total) for _, total, _ in matches}
    if len(total) != 1:
        raise ValueError(f"{path} mixes schedules of different length")
    return int(seed.group(1)), {int(epoch): float(value) for epoch, _, value in matches}, total.pop()


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    return np.array([values[max(0, i - half):i + half + 1].mean() for i in range(len(values))])


def choose(curves: dict[int, dict[int, float]], schedule: int, tolerance: float, window: int):
    epochs = sorted(set.intersection(*(set(points) for points in curves.values())))
    if len(epochs) < 3:
        raise ValueError("need at least three shared validation epochs")
    if epochs[-1] != schedule:
        raise ValueError(f"last shared validation epoch {epochs[-1]} is not the schedule end {schedule}")
    seeds = sorted(curves)
    matrix = np.array([[curves[seed][epoch] for epoch in epochs] for seed in seeds])
    mean = matrix.mean(axis=0)
    smoothed = smooth(mean, window)
    best_index = int(np.argmax(smoothed))  # earliest epoch on ties
    tail = max(3, len(epochs) // 3)
    slope = float(np.polyfit(epochs[-tail:], mean[-tail:], 1)[0]) * 10
    per_seed_best = {seed: epochs[int(np.argmax(row))] for seed, row in zip(seeds, matrix)}
    gap = float(smoothed[best_index] - smoothed[-1])
    if best_index == len(epochs) - 1 and slope > 0:
        decision, budget = "extend_schedule", None
        reason = "the smoothed mean peaks at the last epoch and is still rising"
    elif gap <= tolerance:
        decision, budget = "full_schedule", schedule
        reason = f"the last epoch is within {tolerance:g} of the smoothed maximum (gap {gap:.4f})"
    else:
        decision, budget = "smoothed_peak", epochs[best_index]
        reason = (f"the last epoch trails the smoothed maximum by {gap:.4f}; a shorter run anneals "
                  "its cosine schedule earlier than the validated runs")
    return {
        "decision": decision, "budget": budget, "reason": reason, "schedule_epochs": schedule,
        "seeds": seeds, "tolerance": tolerance, "smoothing_window": window,
        "smoothed_peak_epoch": epochs[best_index], "smoothed_peak": float(smoothed[best_index]),
        "smoothed_last": float(smoothed[-1]), "last_minus_peak": -gap,
        "late_slope_per_10_epochs": slope, "late_points": tail,
        "per_seed_best_epoch": per_seed_best,
        "median_selected_epoch": statistics.median(per_seed_best.values()),
        "seeds_peaking_at_last_epoch": [seed for seed, epoch in per_seed_best.items() if epoch == schedule],
        "curve": [{"epoch": epoch, "mean": float(m), "smoothed": float(s),
                   **{f"seed{seed}": float(v) for seed, v in zip(seeds, matrix[:, index])}}
                  for index, (epoch, m, s) in enumerate(zip(epochs, mean, smoothed))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run directories with history.json and config.yaml")
    parser.add_argument("--logs", nargs="*", type=Path, default=[], help="Slurm logs, one training run each")
    parser.add_argument("--metric", default="score", help="validation key, e.g. score or clean")
    parser.add_argument("--tolerance", type=float, default=.01)
    parser.add_argument("--smooth", type=int, default=3, help="centered moving-average window (odd)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.runs) == bool(args.logs) or args.smooth < 1 or args.smooth % 2 == 0:
        raise SystemExit("give either run directories or --logs, and an odd positive --smooth")
    curves, schedules = {}, set()
    sources = [(curve_from_history, path) for path in args.runs] + [(curve_from_log, path) for path in args.logs]
    for reader, path in sources:
        seed, points, schedule = reader(path, args.metric)
        if seed in curves:
            raise SystemExit(f"duplicate seed {seed}: pass one run per seed")
        curves[seed], schedules = points, schedules | {schedule}
    if len(schedules) != 1:
        raise SystemExit(f"runs use different schedule lengths: {sorted(schedules)}")
    result = {"metric": args.metric, "sources": [str(path) for _, path in sources],
              **choose(curves, schedules.pop(), args.tolerance, args.smooth)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "curve"}, indent=2))


if __name__ == "__main__":
    main()
