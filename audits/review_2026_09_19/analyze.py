"""Read-only analysis of saved runs; writes derived review files beside this script.

Run from any directory with python3 audits/review_2026_09_19/analyze.py.
The core cohort uses the earliest TCN run per method/seed. The ablation cohort
uses the later OAP repetition and the seven no_* methods, matching the job logs.
These are timestamp rules, never score-based selection. Training is not run.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path
import statistics
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
CORE = ["ce", "masked_ce", "generic_supcon", "global_occlusion_supcon", "oap_supcon"]
FAMILIES = ["random_joint", "dynamic_joint", "body_part", "temporal"]


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    runs = []
    by_method_seed = defaultdict(list)
    for p in sorted((ROOT / "results").glob("*/metrics.json")):
        run = json.loads(p.read_text())
        run["path"] = p.parent
        run["config"] = yaml.safe_load((p.parent / "config.yaml").read_text())
        runs.append(run)
        by_method_seed[(run["backbone"], run["method"], run["seed"])].append(run)
    manifests, grouped = [], defaultdict(list)
    for run in runs:
        method = run["method"]
        candidates = by_method_seed[(run["backbone"], method, run["seed"])]
        if run["backbone"] != "tcn":
            cohort = "backbone_pilot"
        elif "__" in method:
            cohort = "representation_pilot"
        elif method in CORE:
            cohort = "core" if run is candidates[0] else "ablation"
        else:
            cohort = "ablation"
        run["cohort"] = cohort
        manifests.append({
            "cohort": cohort, "run_id": run["run_id"], "backbone": run["backbone"],
            "method": method, "seed": run["seed"], "commit": run["git_commit"],
            "dataset_sha256": run["dataset_sha256"], "epochs": run["epochs"],
            "weight_coordinates": run["config"]["pose"]["weight_coordinates"],
            "clip_length": run["config"]["train"].get("clip_length"),
            "parameters": run["parameters"], "train_hours": run["train_hours"],
        })
        for row in run["results"]:
            grouped[(cohort, run["backbone"], method, row["descriptor"], row["protocol"],
                     row["corruption"], row["severity"])].append((run["seed"], row["rank1"]))
    summary = []
    for key, values in sorted(grouped.items()):
        seeds, scores = zip(*values)
        assert len(seeds) == len(set(seeds)), (key, seeds)
        summary.append({
            **dict(zip(["cohort", "backbone", "method", "descriptor", "protocol", "corruption", "severity"], key)),
            "n_seeds": len(seeds), "rank1_mean_pct": 100 * statistics.mean(scores),
            "rank1_sd_pp": 100 * statistics.stdev(scores) if len(scores) > 1 else "",
        })
    write_csv("run_manifest.csv", manifests)
    write_csv("rank1_summary.csv", summary)

    duplicates = []
    for (backbone, method, seed), group in by_method_seed.items():
        if len(group) < 2:
            continue
        a, b = group
        differences = [abs(ra["rank1"] - rb["rank1"]) * 100
                       for ra, rb in zip(a["results"], b["results"])]
        duplicates.append({"backbone": backbone, "method": method, "seed": seed,
                           "first": a["run_id"], "second": b["run_id"],
                           "max_rank1_difference_pp": max(differences)})
    write_csv("repeated_seeds.csv", duplicates)

    # Existing saved outcome files: exploratory paired differences by seed and
    # identity. Do not count multiple corruption realizations as new identities.
    paired = []
    comparisons = [("core", "masked_ce", "core", "oap_supcon"),
                   ("ablation", "oap_supcon", "ablation", "no_complete_partial"),
                   ("ablation", "no_part", "ablation", "oap_supcon")]
    for ca, ma, cb, mb in comparisons:
        for seed in [11, 22, 33, 44, 55]:
            a = next(r for r in runs if (r["cohort"], r["method"], r["seed"], r["backbone"]) == (ca, ma, seed, "tcn"))
            b = next(r for r in runs if (r["cohort"], r["method"], r["seed"], r["backbone"]) == (cb, mb, seed, "tcn"))
            with np.load(a["path"] / "probe_outcomes.npz") as za, np.load(b["path"] / "probe_outcomes.npz") as zb:
                assert np.array_equal(za["probe_indices"], zb["probe_indices"])
                assert np.array_equal(za["probe_labels"], zb["probe_labels"])
                labels = za["probe_labels"]
                for family in FAMILIES:
                    key = f"global_parts_clean_gallery_{family}_0.5"
                    diff = zb[key] - za[key]
                    identity_diff = np.array([diff[labels == ident].mean() for ident in np.unique(labels)])
                    draws = np.random.default_rng(2026).choice(identity_diff, (10000, len(identity_diff))).mean(1)
                    paired.append({"baseline_cohort": ca, "baseline": ma, "candidate_cohort": cb,
                                   "candidate": mb, "seed": seed, "outcome": key,
                                   "difference_pp": float(identity_diff.mean() * 100),
                                   "ci_low_pp": float(np.quantile(draws, .025) * 100),
                                   "ci_high_pp": float(np.quantile(draws, .975) * 100)})
    write_csv("exploratory_paired_bootstrap.csv", paired)

    import torch
    from oap_supcon.augment import anatomical_parts, corrupt, generic_view, speed_perturb, temporal_crop_view
    from oap_supcon.model import build_encoder
    torch.set_num_threads(2)
    x = torch.ones(4, 60, 17, 2)
    visibility = torch.full(x.shape[:3], .8)
    lengths = torch.full((4,), 60)
    gen = lambda: torch.Generator().manual_seed(11)
    diagnostics = {}
    for name, fn in [
        ("generic_no_noise_scale_one", lambda: generic_view(x, visibility, gen(), noise=0, scale_range=(1, 1))),
        ("crop_full", lambda: temporal_crop_view(x, visibility, gen(), 1, 1, lengths)),
        ("speed_one", lambda: speed_perturb(x, visibility, gen(), (1, 1), lengths)),
        ("corrupt_epsilon", lambda: corrupt(x, visibility, "random_joint", 1e-10, gen(), lengths)),
        ("corrupt_zero", lambda: corrupt(x, visibility, "random_joint", 0, gen(), lengths)),
    ]:
        y, mask = fn()
        diagnostics[name] = {"mean_absolute_visible_coordinate": float(y[mask > 0].abs().mean()),
                             "removed_joint_frames": int((mask == 0).sum())}
    torch.manual_seed(11)
    model = build_encoder("tcn", 2, 17, 4).eval()
    v = torch.ones(1, 60, 17)
    v[:, :, [5, 7, 9]] = 0
    with torch.no_grad():
        output = model(torch.randn(1, 60, 17, 2) * v.unsqueeze(-1), v)
    diagnostics["absent_left_arm"] = {"reliability": float(output["reliability"][0, 1]),
                                      "descriptor_norm": float(output["parts"][0, 1].norm())}
    severity_rows = []
    for severity in [.1, .3, .5, .7]:
        counts = []
        parts = anatomical_parts(17)
        for order in itertools.permutations(range(5)):
            selected = set()
            for part in order:
                selected.update(parts[part])
                if len(selected) >= max(1, round(severity * 17)):
                    break
            counts.append(len(selected) / 17)
        severity_rows.append({"nominal": severity, "body_part_actual_mean": statistics.mean(counts),
                              "body_part_actual_min": min(counts), "body_part_actual_max": max(counts),
                              "dynamic_joint_interior_expected": 3 * severity**2 - 2 * severity**3})
    diagnostics["severity_calibration"] = severity_rows
    with np.load(ROOT / "data/casia_b_pose/processed/dataset.npz") as z:
        v = z["visibility"]
        lengths = z["frame_counts"]
        total = count = 0
        for start in range(0, len(v), 256):
            block = v[start:start + 256]
            real = np.arange(v.shape[1])[None, :] < lengths[start:start + 256, None]
            real_values = block[real]
            total += float(real_values.sum(dtype=np.float64))
            count += real_values.size
        diagnostics["source_mean_confidence"] = total / count
        diagnostics["data_splits"] = dict(zip(*[a.tolist() for a in np.unique(z["split"], return_counts=True)]))
    (HERE / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/oap-review-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    series = [("core", "ce", "global", "CE · global", "#526477"),
              ("core", "masked_ce", "global_parts", "Masked CE · global + parts", "#cb7922"),
              ("core", "oap_supcon", "global_parts", "OAP · global + parts", "#8055a5"),
              ("ablation", "no_complete_partial", "global_parts", "Two partial views · global + parts", "#16826c"),
              ("backbone_pilot", "oap_supcon", "global", "ST-GCN OAP pilot · global (one seed)", "#2572b8")]
    for ax, family in zip(axes.flat, FAMILIES):
        for cohort, method, desc, label, color in series:
            rows = sorted([r for r in summary if r["cohort"] == cohort and r["method"] == method
                           and r["descriptor"] == desc and r["protocol"] == "clean_gallery"
                           and r["corruption"] == family], key=lambda r: r["severity"])
            xs = np.array([r["severity"] for r in rows])
            means = np.array([r["rank1_mean_pct"] for r in rows])
            sd = np.array([r["rank1_sd_pp"] or 0 for r in rows])
            ax.plot(xs, means, marker="o", markersize=4, color=color, label=label,
                    linestyle="--" if cohort == "backbone_pilot" else "-")
            if cohort != "backbone_pilot":
                ax.fill_between(xs, means - sd, means + sd, color=color, alpha=.10)
        ax.set_title(family.replace("_", " ").capitalize(), loc="left", fontweight="bold")
        ax.axhline(2, color="#aaaaaa", linewidth=.8, linestyle=":")
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([0, .1, .3, .5, .7])
        ax.set_ylim(0, 65)
        ax.set_xlabel("Nominal corruption severity")
        ax.set_ylabel("Rank-1 (%)")
    fig.suptitle("CASIA-B: clean gallery, pooled across views; identical-view matches excluded", fontsize=14, y=.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .025))
    fig.text(.5, .006, "Solid lines: five seeds, shaded ±1 SD. Dashed line: one pilot seed. Descriptor choices shown explicitly; see report for matched comparisons.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .95))
    fig.savefig(HERE / "robustness.png", dpi=180)
    fig.savefig(HERE / "robustness.pdf")
    print(json.dumps({"runs": len(runs), "cohorts": {c: sum(r['cohort'] == c for r in runs) for c in sorted(set(r['cohort'] for r in runs))},
                      "diagnostics": diagnostics, "repeated_seeds": duplicates}, indent=2))


if __name__ == "__main__":
    main()
