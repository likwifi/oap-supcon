#!/usr/bin/env python3
"""Rebuild IJCV development uncertainty and Fig4 from saved outcomes; no training.

Run from any directory with Python, NumPy, PyYAML, and Matplotlib installed.
Historical result files are read only. The bootstrap treats model seeds and
identities as crossed factors, because the same identities occur in every seed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_factorial as factorial
import analyze_descriptor_controls as controls

SAMPLES = 100_000
BOOTSTRAP_SEED = 20261002
ENDPOINTS = ("clean", "robust", "severity_05", "occluded")


def crossed_bootstrap(blocks, name):
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.shape != (3, 12) or not np.isfinite(blocks).all():
        raise ValueError(f"unexpected seed/identity blocks for {name}: {blocks.shape}")
    # A stable per-comparison RNG makes results independent of iteration order.
    salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, salt]))
    seeds = rng.integers(0, 3, size=(SAMPLES, 3))
    identities = rng.integers(0, 12, size=(SAMPLES, 12))
    draws = blocks[seeds[:, :, None], identities[:, None, :]].mean(axis=(1, 2))
    return {
        "difference": float(blocks.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "seed_differences": blocks.mean(axis=1).tolist(),
    }


def validate_identity_order(runs):
    reference = None
    for directory in runs:
        with np.load(directory / "probe_outcomes.npz") as archive:
            current = (archive["probe_indices"], archive["probe_labels"])
            if reference is None:
                reference = tuple(x.copy() for x in current)
            if not all(np.array_equal(a, b) for a, b in zip(reference, current)):
                raise ValueError(f"probe alignment differs in {directory}")
    if len(np.unique(reference[1])) != 12:
        raise ValueError("expected 12 development identities")


def draw_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (ROOT / "results_v2_factorial/summary.csv").open() as handle:
        training_rows = list(csv.DictReader(handle))
    with (ROOT / "results_descriptor_controls/summary.csv").open() as handle:
        matching_rows = list(csv.DictReader(handle))
    families = factorial.FAMILIES
    titles = ["Random joint-frame", "Dynamic joint (held out)", "Whole part", "Temporal interruption"]
    severities = [0, 0.1, 0.3, 0.5, 0.7]
    configurations = [
        ("Metric baseline / global", "metric_baseline", "#264b75", "o", "-"),
        ("Global SupCon / global", "metric_global_supcon", "#ad761a", "s", "--"),
        ("Part SupCon / global", "metric_part_supcon", "#b34c46", "^", "-."),
        ("Metric baseline / raw parts", "raw_reliable", "#258366", "D", "-"),
    ]
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1), sharex=True, sharey=True)
    numerical = {}
    for ax, family, title in zip(axes.flat, families, titles):
        numerical[family] = {}
        for label, method, color, marker, linestyle in configurations:
            means, deviations = [], []
            for severity in severities:
                rows = matching_rows if method == "raw_reliable" else training_rows
                selected = [r for r in rows if r["protocol"] == "clean_gallery"
                            and r["corruption"] == family and float(r["severity"]) == severity
                            and ((method == "raw_reliable" and r["descriptor"] == method
                                  and float(r["fusion_weight"]) == .5)
                                 or (method != "raw_reliable" and r["method"] == method
                                     and r["descriptor"] == "global"))]
                if len(selected) != 1 or int(selected[0]["rank1_count"]) != 3:
                    raise ValueError(f"incomplete curve cell: {method}, {family}, {severity}")
                means.append(100 * float(selected[0]["rank1_mean"]))
                deviations.append(100 * float(selected[0]["rank1_std"]))
            mean, sd = np.array(means), np.array(deviations)
            ax.plot(severities, mean, label=label, color=color, marker=marker,
                    linestyle=linestyle, linewidth=1.5, markersize=3.5)
            ax.fill_between(severities, mean - sd, mean + sd, color=color, alpha=.10, linewidth=0)
            numerical[family][method] = {"mean_percent": means, "sd_percent": deviations}
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks(severities)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linewidth=.4, alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Nominal corruption severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("Rank-1 (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(HERE / "Fig4.pdf", bbox_inches="tight")
    plt.close(fig)
    return numerical


def main():
    runs = factorial.discover_runs(ROOT / "results_v2_factorial")
    control_runs, _ = controls.discover(ROOT / "results_descriptor_controls", factorial.SEEDS)
    validate_identity_order(list(runs.values()) + list(control_runs.values()))
    paired = {}
    for descriptor in factorial.DESCRIPTORS:
        for method in factorial.METHODS[1:]:
            key = f"{method}/{descriptor}"
            paired[key] = {
                endpoint: crossed_bootstrap(
                    factorial.between_method_blocks(runs, method, descriptor, endpoint),
                    f"{key}/{endpoint}",
                ) for endpoint in ENDPOINTS
            }
    descriptor_effects = {}
    for descriptor in ("raw_reliable", "raw_uniform", "frozen_head_reliable"):
        descriptor_effects[descriptor] = {
            endpoint: crossed_bootstrap(
                controls.paired_blocks(control_runs, factorial.SEEDS, descriptor, .5, endpoint),
                f"{descriptor}/{endpoint}",
            ) for endpoint in ENDPOINTS
        }
    descriptor_effects["raw_reliable_minus_uniform"] = {
        endpoint: crossed_bootstrap(
            controls.paired_blocks(control_runs, factorial.SEEDS, "raw_reliable", .5, endpoint,
                                   reference_descriptor="raw_uniform", reference_weight=.5),
            f"raw_reliable_minus_uniform/{endpoint}",
        ) for endpoint in ENDPOINTS
    }
    manifest = []
    for directory in sorted(list(runs.values()) + list(control_runs.values())):
        m = json.loads((directory / "metrics.json").read_text())
        manifest.append({
            "directory": str(directory.relative_to(ROOT)),
            "source_sha256": m.get("source_sha256"),
            "dataset_sha256": m.get("dataset_sha256"),
            "artifacts_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                 for name in ("metrics.json", "probe_outcomes.npz")},
        })
    result = {
        "bootstrap": {"type": "crossed", "samples": SAMPLES, "seed": BOOTSTRAP_SEED,
                      "training_seeds": factorial.SEEDS, "identity_count": 12},
        "against_baseline": paired,
        "descriptor_effects": descriptor_effects,
        "checkpoint_diagnostics": {
            method: {
                str(seed): {
                    "selected_epoch": (m := json.loads((runs[(method, seed)] / "metrics.json").read_text()))["selected_epoch"],
                    "selected_epoch_part_variance": float(np.mean(
                        json.loads((runs[(method, seed)] / "history.json").read_text())
                        [m["selected_epoch"] - 1]["part_embedding_variance"])),
                } for seed in factorial.SEEDS
            } for method in factorial.METHODS
        },
        "curves": draw_curves(),
        "manifest": manifest,
    }
    (HERE / "evidence_crossed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"part_supcon": paired["metric_part_supcon/global"]["robust"],
                      "raw_matching": descriptor_effects["raw_reliable"]["robust"]}, indent=2))


if __name__ == "__main__":
    main()
