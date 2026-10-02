#!/usr/bin/env python3
"""Regenerate paper tables from the completed Benchmark V2 factorial.

The script deliberately avoids pandas so it can run in the project environment
without compiled dataframe dependencies.  It validates the 6 x 3 design,
summarises the aggregate metrics, and performs a paired hierarchical bootstrap
over training seed and validation identity using the saved probe outcomes.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


METHODS = [
    "metric_baseline",
    "metric_global_supcon",
    "metric_part_supcon",
    "metric_part_triplet",
    "oap_v2",
    "oap_v2_ungated",
]
SEEDS = [11, 22, 33]
DESCRIPTORS = ["global", "reliable_parts"]
FAMILIES = ["random_joint", "dynamic_joint", "body_part", "temporal"]
SEVERITIES = ["0.1", "0.3", "0.5", "0.7"]
DISPLAY = {
    "metric_baseline": "Metric baseline",
    "metric_global_supcon": "Global SupCon",
    "metric_part_supcon": "Part SupCon",
    "metric_part_triplet": "Part triplet",
    "oap_v2": "Global + part SupCon",
    "oap_v2_ungated": "Global + part SupCon (ungated)",
    "global": "Global",
    "reliable_parts": "Reliable-parts",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=Path("results_v2_factorial"))
    parser.add_argument("--output", type=Path, default=Path("Paper/generated"))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261001)
    return parser.parse_args()


def discover_runs(root: Path):
    runs = {}
    required = ("config.yaml", "history.json", "metrics.json", "probe_outcomes.npz")
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            raise SystemExit(f"{directory} is incomplete: missing {', '.join(missing)}")
        config = yaml.safe_load((directory / "config.yaml").read_text())
        key = (config["method_name"], int(config["seed"]))
        if key in runs:
            raise SystemExit(f"duplicate run for {key}: {runs[key]} and {directory}")
        runs[key] = directory
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if set(runs) != expected:
        missing = sorted(expected.difference(runs))
        extra = sorted(set(runs).difference(expected))
        raise SystemExit(f"factorial mismatch; missing={missing}, extra={extra}")
    return runs


def read_summary(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 684:
        raise SystemExit(f"expected 684 aggregate rows, found {len(rows)}")
    if {row["rank1_count"] for row in rows} != {"3"}:
        raise SystemExit("every aggregate cell must contain all three seeds")
    return rows


def selected(rows, method, descriptor, predicate, field="rank1_mean"):
    return [
        float(row[field])
        for row in rows
        if row["method"] == method and row["descriptor"] == descriptor and predicate(row)
    ]


def aggregate_results(rows):
    result = {}
    for method in METHODS:
        for descriptor in DESCRIPTORS:
            clean = selected(
                rows,
                method,
                descriptor,
                lambda row: row["protocol"] == "clean_gallery"
                and row["corruption"] == "body_part"
                and float(row["severity"]) == 0,
            )
            clean_sd = selected(
                rows,
                method,
                descriptor,
                lambda row: row["protocol"] == "clean_gallery"
                and row["corruption"] == "body_part"
                and float(row["severity"]) == 0,
                "rank1_std",
            )
            robust = selected(
                rows,
                method,
                descriptor,
                lambda row: row["protocol"] == "clean_gallery"
                and row["corruption"] in FAMILIES
                and float(row["severity"]) > 0,
            )
            severity_05 = selected(
                rows,
                method,
                descriptor,
                lambda row: row["protocol"] == "clean_gallery"
                and row["corruption"] in FAMILIES
                and float(row["severity"]) == 0.5,
            )
            occluded = selected(
                rows,
                method,
                descriptor,
                lambda row: row["protocol"] == "occluded_gallery"
                and row["corruption"] in FAMILIES
                and float(row["severity"]) > 0,
            )
            official = selected(
                rows,
                method,
                descriptor,
                lambda row: row["protocol"] == "single_view_gallery"
                and row["corruption"].startswith("official_condition:"),
            )
            hours = selected(rows, method, descriptor, lambda _row: True, "train_hours_mean")
            if not all((clean, clean_sd, robust, severity_05, occluded, official, hours)):
                raise SystemExit(f"missing aggregate cells for {(method, descriptor)}")
            result[(method, descriptor)] = {
                "clean": float(np.mean(clean)),
                "clean_sd": float(np.mean(clean_sd)),
                "robust": float(np.mean(robust)),
                "severity_05": float(np.mean(severity_05)),
                "occluded": float(np.mean(occluded)),
                "official": float(np.mean(official)),
                "train_hours": float(np.mean(hours)),
            }
    return result


def endpoint_keys(descriptor, endpoint):
    if endpoint == "clean":
        return [f"{descriptor}_clean_gallery_body_part_0"]
    if endpoint == "robust":
        return [
            f"{descriptor}_clean_gallery_{family}_{severity}"
            for family in FAMILIES
            for severity in SEVERITIES
        ]
    if endpoint == "severity_05":
        return [f"{descriptor}_clean_gallery_{family}_0.5" for family in FAMILIES]
    if endpoint == "occluded":
        return [
            f"{descriptor}_occluded_gallery_{family}_{severity}"
            for family in FAMILIES
            for severity in SEVERITIES
        ]
    raise ValueError(endpoint)


def averaged_outcome(archive, keys):
    missing = set(keys).difference(archive.files)
    if missing:
        raise SystemExit(f"probe archive is missing keys: {sorted(missing)}")
    return np.nanmean(np.stack([archive[key] for key in keys]), axis=0)


def between_method_blocks(runs, candidate, descriptor, endpoint):
    blocks = []
    keys = endpoint_keys(descriptor, endpoint)
    for seed in SEEDS:
        with np.load(runs[("metric_baseline", seed)] / "probe_outcomes.npz") as baseline, np.load(
            runs[(candidate, seed)] / "probe_outcomes.npz"
        ) as current:
            if not np.array_equal(baseline["probe_indices"], current["probe_indices"]):
                raise SystemExit(f"probe ordering mismatch for {candidate}, seed {seed}")
            if not np.array_equal(baseline["probe_labels"], current["probe_labels"]):
                raise SystemExit(f"probe labels mismatch for {candidate}, seed {seed}")
            labels = baseline["probe_labels"]
            before = averaged_outcome(baseline, keys)
            after = averaged_outcome(current, keys)
            identities = np.unique(labels)
            blocks.append(
                [float(np.nanmean(after[labels == identity] - before[labels == identity])) for identity in identities]
            )
    return np.asarray(blocks)


def between_descriptor_blocks(runs, method, endpoint):
    blocks = []
    global_keys = endpoint_keys("global", endpoint)
    part_keys = endpoint_keys("reliable_parts", endpoint)
    for seed in SEEDS:
        with np.load(runs[(method, seed)] / "probe_outcomes.npz") as archive:
            labels = archive["probe_labels"]
            before = averaged_outcome(archive, global_keys)
            after = averaged_outcome(archive, part_keys)
            identities = np.unique(labels)
            blocks.append(
                [float(np.nanmean(after[labels == identity] - before[labels == identity])) for identity in identities]
            )
    return np.asarray(blocks)


def hierarchical_bootstrap(blocks, samples, rng):
    seed_count, identity_count = blocks.shape
    seed_draws = rng.integers(0, seed_count, size=(samples, seed_count))
    identity_draws = rng.integers(
        0, identity_count, size=(samples, seed_count, identity_count)
    )
    draws = np.take_along_axis(blocks[seed_draws], identity_draws, axis=2).mean(axis=(1, 2))
    return {
        "difference": float(blocks.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "probability_positive": float(np.mean(draws > 0)),
        "seed_blocks": int(seed_count),
        "identity_blocks_per_seed": int(identity_count),
    }


def paired_results(runs, samples, seed):
    rng = np.random.default_rng(seed)
    result = {"against_baseline": {}, "descriptor_effect": {}}
    endpoints = ("clean", "robust", "severity_05", "occluded")
    for candidate in METHODS[1:]:
        result["against_baseline"][candidate] = {}
        for descriptor in DESCRIPTORS:
            result["against_baseline"][candidate][descriptor] = {
                endpoint: hierarchical_bootstrap(
                    between_method_blocks(runs, candidate, descriptor, endpoint), samples, rng
                )
                for endpoint in endpoints
            }
    for method in METHODS:
        result["descriptor_effect"][method] = {
            endpoint: hierarchical_bootstrap(
                between_descriptor_blocks(runs, method, endpoint), samples, rng
            )
            for endpoint in endpoints
        }
    return result


def history_results(runs):
    rows = {}
    for method in METHODS:
        seeds = []
        cosines = defaultdict(list)
        diagnostic_epochs = 0
        for seed in SEEDS:
            history = json.loads((runs[(method, seed)] / "history.json").read_text())
            if len(history) != 160:
                raise SystemExit(f"{method}, seed {seed} has {len(history)} epochs")
            metrics = json.loads((runs[(method, seed)] / "metrics.json").read_text())
            tail = history[-20:]
            seeds.append(
                {
                    "part_variance": float(
                        np.mean([np.mean(epoch["part_embedding_variance"]) for epoch in tail])
                    ),
                    "part_reliability": float(
                        np.mean([np.mean(epoch["part_reliability"]) for epoch in tail])
                    ),
                    "part_presence": float(
                        np.mean([np.mean(epoch["part_presence_rate"]) for epoch in tail])
                    ),
                    "selected_epoch": int(metrics["selected_epoch"]),
                }
            )
            for epoch in history:
                diagnostics = epoch.get("loss_gradient_cosine") or {}
                if diagnostics:
                    diagnostic_epochs += 1
                for name, value in diagnostics.items():
                    cosines[name].append(float(value))
        rows[method] = {
            key: float(np.mean([row[key] for row in seeds]))
            for key in ("part_variance", "part_reliability", "part_presence", "selected_epoch")
        }
        rows[method]["cosines"] = {
            key: {"mean": float(np.mean(values)), "count": len(values)}
            for key, values in sorted(cosines.items())
        }
        rows[method]["diagnostic_epochs"] = diagnostic_epochs
    return rows


def tex_escape(value):
    return str(value).replace("_", r"\_")


def percent(value):
    return f"{100 * value:.2f}"


def difference_cell(result):
    low, high = result["ci95"]
    return f"{100 * result['difference']:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]"


def write_factorial_table(path, aggregate):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{CASIA-B validation Rank-1 (\%). Clean is reported as mean $\pm$ seed standard deviation. Corrupted-probe columns average the four corruption families; `All $>0$' averages severities 0.1, 0.3, 0.5, and 0.7. Occluded gallery averages the same positive-severity grid. Single-view averages NM, BG, and CL under the CASIA-B single-view-gallery protocol. Every cell uses three training seeds.}",
        r"\label{tab:factorial-results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Training objective & Descriptor & Clean & Clean gallery, all $>0$ & Clean gallery, $\rho=0.5$ & Occluded gallery, all $>0$ & Single-view & Train (h) \\",
        r"\midrule",
    ]
    for method_index, method in enumerate(METHODS):
        for descriptor_index, descriptor in enumerate(DESCRIPTORS):
            values = aggregate[(method, descriptor)]
            method_cell = DISPLAY[method] if descriptor_index == 0 else ""
            lines.append(
                f"{method_cell} & {DISPLAY[descriptor]} & "
                f"{percent(values['clean'])} $\\pm$ {percent(values['clean_sd'])} & "
                f"{percent(values['robust'])} & {percent(values['severity_05'])} & "
                f"{percent(values['occluded'])} & {percent(values['official'])} & "
                f"{values['train_hours']:.2f} \\\\"
            )
        if method_index != len(METHODS) - 1:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(lines))


def write_paired_table(path, paired):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired Rank-1 difference from the metric baseline in percentage points (hierarchical 95\% bootstrap interval over seed and validation identity). Negative values favor the baseline. Corrupted endpoints average the stated corruption cells within each probe before resampling.}",
        r"\label{tab:paired-results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Training objective & Descriptor & Clean & Clean gallery, all $>0$ & Clean gallery, $\rho=0.5$ & Occluded gallery, all $>0$ \\",
        r"\midrule",
    ]
    for method_index, method in enumerate(METHODS[1:]):
        for descriptor_index, descriptor in enumerate(DESCRIPTORS):
            result = paired["against_baseline"][method][descriptor]
            method_cell = DISPLAY[method] if descriptor_index == 0 else ""
            lines.append(
                f"{method_cell} & {DISPLAY[descriptor]} & "
                f"{difference_cell(result['clean'])} & {difference_cell(result['robust'])} & "
                f"{difference_cell(result['severity_05'])} & {difference_cell(result['occluded'])} \\\\"
            )
        if method_index != len(METHODS[1:]) - 1:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(lines))


def write_mechanism_table(path, history):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training diagnostics averaged over three seeds. Part variance is the mean coordinate-wise variance of normalized part embeddings over the final 20 epochs. Cosines are measured every five epochs at the shared paired-view feature tensor; $n$ is the number of finite diagnostics out of 96 scheduled measurements.}",
        r"\label{tab:mechanisms}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Objective & Part variance ($\times 10^{-3}$) & Part--global cosine & Part--triplet cosine & Selected epoch \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = history[method]
        cosines = row["cosines"]
        global_part = cosines.get("global__part")
        if "part__triplet" in cosines:
            part_triplet = cosines["part__triplet"]
        else:
            part_triplet = cosines.get("part_triplet__triplet")
        gp = "--" if global_part is None else f"{global_part['mean']:.3f} ({global_part['count']})"
        pt = "--" if part_triplet is None else f"{part_triplet['mean']:.3f} ({part_triplet['count']})"
        lines.append(
            f"{DISPLAY[method]} & {1000 * row['part_variance']:.3f} & {gp} & {pt} & "
            f"{row['selected_epoch']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines))


def write_paired_csv(path, paired):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["comparison", "method", "descriptor", "endpoint", "difference", "ci95_low", "ci95_high", "probability_positive"]
        )
        for method, descriptors in paired["against_baseline"].items():
            for descriptor, endpoints in descriptors.items():
                for endpoint, result in endpoints.items():
                    writer.writerow(
                        [
                            "against_metric_baseline",
                            method,
                            descriptor,
                            endpoint,
                            result["difference"],
                            *result["ci95"],
                            result["probability_positive"],
                        ]
                    )
        for method, endpoints in paired["descriptor_effect"].items():
            for endpoint, result in endpoints.items():
                writer.writerow(
                    [
                        "reliable_parts_minus_global",
                        method,
                        "reliable_parts",
                        endpoint,
                        result["difference"],
                        *result["ci95"],
                        result["probability_positive"],
                    ]
                )


def main():
    args = parse_args()
    runs = discover_runs(args.results)
    rows = read_summary(args.results / "summary.csv")
    aggregate = aggregate_results(rows)
    paired = paired_results(runs, args.bootstrap_samples, args.bootstrap_seed)
    history = history_results(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    write_factorial_table(args.output / "factorial_results.tex", aggregate)
    write_paired_table(args.output / "paired_differences.tex", paired)
    write_mechanism_table(args.output / "mechanism_results.tex", history)
    write_paired_csv(args.output / "paired_differences.csv", paired)
    payload = {
        "design": {"methods": METHODS, "seeds": SEEDS, "runs": len(runs)},
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed},
        "aggregate": {f"{method}/{descriptor}": values for (method, descriptor), values in aggregate.items()},
        "paired": paired,
        "history": history,
    }
    (args.output / "factorial_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"validated {len(runs)} runs and wrote analysis to {args.output}")


if __name__ == "__main__":
    main()
