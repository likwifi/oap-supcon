#!/usr/bin/env python3
"""Plot synthetic-corruption accuracy against the evidence actually removed.

Nominal severity is not the removed fraction. Body-part masks remove whole
anatomical groups (0.7 removes every body joint and keeps the five face
keypoints), dynamic masks are temporally smoothed, and temporal gaps round to
whole frames. Every clean-gallery evaluation mask is regenerated exactly as
`_evaluate` drew it, from the recorded corruption seed, family order,
realization and probe index, and paired with the saved per-probe outcomes.

The removed fraction is the share of a probe's observed joint-frames that the
mask hid, averaged over the mask realizations whose outcomes were averaged.
Cell rows are exact; per-probe bins pool severities and use that average.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_runs import AP_PREFIX, discover  # noqa: E402
from oap_supcon.augment import corrupt  # noqa: E402
from oap_supcon.data import PoseData, development_split, resolve_dataset_path  # noqa: E402

PROTOCOL_FIELDS = ("corruption_seed", "corruption_families", "severities", "corruption_realizations", "clip_length")


def removed_fractions(data: PoseData, probes: np.ndarray, evaluation: dict) -> dict:
    """{(family, severity): [probe] mean fraction of observed joint-frames hidden}."""
    if evaluation.get("clip_length"):
        raise ValueError("mask regeneration assumes full-sequence evaluation (clip_length null)")
    base = int(evaluation.get("corruption_seed", 1729))
    realizations = int(evaluation["corruption_realizations"])
    fractions = {}
    for family_index, family in enumerate(evaluation["corruption_families"]):
        for severity in (float(value) for value in evaluation["severities"]):
            if severity <= 0:
                continue
            total = np.zeros(len(probes))
            for realization in range(realizations):
                seed = base + family_index * 10_000 + realization
                for position, original in enumerate(probes):
                    length = int(data.frame_counts[original])
                    visibility = torch.from_numpy(np.ascontiguousarray(data.visibility[original, :length]))[None]
                    generator = torch.Generator().manual_seed(seed + int(original) * 1_000_003)
                    x = torch.zeros((*visibility.shape, data.x.shape[-1]))
                    _, mask = corrupt(x, visibility, family, severity, generator,
                                      torch.tensor([length], dtype=torch.long))
                    observed = visibility > 0
                    total[position] += float((observed & (mask == 0)).sum() / observed.sum().clamp_min(1))
            fractions[(family, severity)] = total / realizations
    return fractions


def write_csv(path: Path, rows: list[dict]):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot(path: Path, rows: list[dict], families: list[str], series: list[tuple[str, str]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(2, len(families))
    figure, axes = plt.subplots((len(families) + 1) // 2, columns, figsize=(7.0, 2.6 * ((len(families) + 1) // 2)),
                                sharex=True, sharey=True, squeeze=False)
    for axis, family in zip(axes.flat, families):
        for method, descriptor in series:
            cells = sorted((r for r in rows if r["family"] in (family, "clean") and r["method"] == method
                            and r["descriptor"] == descriptor), key=lambda r: r["removed_fraction_mean"])
            if not cells:
                continue
            x = [r["removed_fraction_mean"] for r in cells]
            axis.plot(x, [r["rank1_mean"] for r in cells], marker="o", markersize=3, label=f"{method} / {descriptor}")
            for r in cells:
                if r["family"] != "clean":
                    axis.annotate(f"{r['nominal_severity']:g}", (r["removed_fraction_mean"], r["rank1_mean"]),
                                  textcoords="offset points", xytext=(2, 3), fontsize=6, color="0.4")
        axis.set_title(family.replace("_", " "), fontsize=9)
        axis.grid(alpha=.3)
    for axis in axes[-1]:
        axis.set_xlabel("observed joint-frames actually removed")
    for axis in axes[:, 0]:
        axis.set_ylabel("Rank-1")
    for axis in list(axes.flat)[len(families):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=6)
    figure.text(.99, .01, "point labels: nominal severity", ha="right", fontsize=6, color="0.4")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def analyze(root: Path, output: Path, dataset_name: str, methods: list[str], seeds: list[int], *,
            run_names=None, dataset: Path | None = None, bin_width: float = .1,
            plot_descriptors=("global", "raw_reliable")):
    runs = discover(root, dataset_name, methods, seeds, run_names)
    first_directory, cfg, first = runs[(methods[0], seeds[0])]
    evaluation = cfg["evaluation"]
    for key, (_, current, _) in runs.items():
        if any(current["evaluation"].get(field) != evaluation.get(field) for field in PROTOCOL_FIELDS):
            raise ValueError(f"corruption protocol differs for {key}")
    if dataset is None:
        recorded = Path(cfg.get("dataset_path", ""))
        dataset = recorded if recorded.is_file() else resolve_dataset_path(ROOT, cfg["dataset"])
    data, _ = development_split(PoseData.load(dataset), cfg)
    if data.checksum() != first["dataset_sha256"]:
        raise ValueError("dataset or its validation split does not match the runs; pass --dataset")
    with np.load(first_directory / "probe_outcomes.npz", allow_pickle=False) as archive:
        probes = archive["probe_indices"]
    fractions = removed_fractions(data, probes, evaluation)
    families = list(evaluation["corruption_families"])
    descriptors = list(evaluation["descriptors"])
    cells, binned = [], []
    edges = np.arange(0, 1 + bin_width, bin_width)
    for method in methods:
        for descriptor in descriptors:
            outcomes = {}
            for seed in seeds:
                with np.load(runs[(method, seed)][0] / "probe_outcomes.npz", allow_pickle=False) as archive:
                    if not np.array_equal(archive["probe_indices"], probes):
                        raise ValueError(f"probe ordering differs for {method} seed {seed}")
                    for (family, severity) in [(families[0], 0.0), *fractions]:
                        key = f"{descriptor}_clean_gallery_{family}_{severity:g}"
                        outcomes.setdefault((family, severity, "rank1"), []).append(archive[key])
                        if AP_PREFIX + key in archive.files:
                            outcomes.setdefault((family, severity, "map"), []).append(archive[AP_PREFIX + key])
            clean = np.stack(outcomes[(families[0], 0.0, "rank1")])
            clean_map = outcomes.get((families[0], 0.0, "map"))
            cells.append({"method": method, "descriptor": descriptor, "family": "clean", "nominal_severity": 0.0,
                          "removed_fraction_mean": 0.0, "removed_fraction_sd": 0.0,
                          "rank1_mean": float(clean.mean()),
                          "map_mean": float(np.mean(clean_map)) if clean_map else None,
                          **{f"rank1_seed{s}": float(v) for s, v in zip(seeds, clean.mean(1))}})
            for family in families:
                pooled_fraction, pooled_correct = [np.zeros(len(probes))], [clean.mean(0)]
                for (current, severity), removed in fractions.items():
                    if current != family:
                        continue
                    correct = np.stack(outcomes[(family, severity, "rank1")])
                    ap = outcomes.get((family, severity, "map"))
                    cells.append({"method": method, "descriptor": descriptor, "family": family,
                                  "nominal_severity": severity, "removed_fraction_mean": float(removed.mean()),
                                  "removed_fraction_sd": float(removed.std()), "rank1_mean": float(correct.mean()),
                                  "map_mean": float(np.mean(ap)) if ap else None,
                                  **{f"rank1_seed{s}": float(v) for s, v in zip(seeds, correct.mean(1))}})
                    pooled_fraction.append(removed)
                    pooled_correct.append(correct.mean(0))
                fraction, correct = np.concatenate(pooled_fraction), np.concatenate(pooled_correct)
                assignment = np.clip(np.digitize(fraction, edges[1:-1], right=False), 0, len(edges) - 2)
                for number in range(len(edges) - 1):
                    members = assignment == number
                    if members.any():
                        binned.append({"method": method, "descriptor": descriptor, "family": family,
                                       "removed_fraction_low": float(edges[number]),
                                       "removed_fraction_high": float(edges[number + 1]),
                                       "probe_cells": int(members.sum()), "rank1_mean": float(correct[members].mean())})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "actual_missingness_cells.csv", cells)
    write_csv(output / "actual_missingness_bins.csv", binned)
    series = [(method, descriptor) for method in methods for descriptor in plot_descriptors if descriptor in descriptors]
    plot(output / "actual_missingness.png", cells, families, series)
    nominal = {f"{family}:{severity:g}": {"mean": float(removed.mean()), "min": float(removed.min()),
                                          "max": float(removed.max())}
               for (family, severity), removed in fractions.items()}
    result = {"dataset": dataset_name, "evaluation_split": first["evaluation_split"], "methods": methods,
              "seeds": seeds, "probes": int(len(probes)), "dataset_sha256": first["dataset_sha256"],
              "protocol": {field: evaluation.get(field) for field in PROTOCOL_FIELDS},
              "removed_fraction_by_nominal_severity": nominal}
    (output / "actual_missingness.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--dataset", required=True, help="dataset name, e.g. casia_b_pose or gait3d")
    parser.add_argument("--methods", required=True)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--bin-width", type=float, default=.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.results, args.output or args.results / "actual_missingness", args.dataset,
                     args.methods.split(","), [int(v) for v in args.seeds.split(",")],
                     run_names=args.runs, dataset=args.dataset_path, bin_width=args.bin_width)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
