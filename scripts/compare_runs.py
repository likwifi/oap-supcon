#!/usr/bin/env python3
"""Compare methods and descriptors from one matched cohort, with Rank-1 and mAP.

A cohort is one dataset, evaluation stage, data checksum and source revision,
with every method trained on the same seeds. Every endpoint reports Rank-1 and,
when the runs stored per-probe average precision (`ap__` keys), mAP. Intervals
are crossed seed x identity bootstraps of identity-level means; differences
pair methods by seed index and probe identity.

Endpoints use the pooled-gallery probe outcomes: `clean` is the uncorrupted
probe set, `robust` averages every positive-severity clean-gallery corruption
cell within each probe, `occluded` does the same with corrupted galleries, and
`family:<name>` restricts `robust` to one corruption family. Official
single-view-gallery CASIA-B scores are matrices over views, not per-probe
outcomes, and are summarized from metrics.json without intervals.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_stats import crossed_draws  # noqa: E402

AP_PREFIX = "ap__"
DESCRIPTOR_CONTRASTS = (("raw_reliable", "global"), ("raw_uniform", "global"), ("raw_reliable", "raw_uniform"))


def discover(root: Path, dataset: str, methods: list[str], seeds: list[int], run_names=None):
    """Return {(method, seed): (directory, config, metrics)} for one matched cohort."""
    paths = (sorted(root.glob("*/config.yaml")) if run_names is None
             else [root / name / "config.yaml" for name in run_names])
    runs, provenance = {}, set()
    for path in paths:
        if not path.exists():
            raise ValueError(f"selected run has no config.yaml: {path.parent}")
        config = yaml.safe_load(path.read_text())
        method = config.get("method_name")
        if config.get("dataset", {}).get("name") != dataset or method not in methods:
            continue
        directory = path.parent
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists() or not (directory / "probe_outcomes.npz").exists():
            if run_names is not None:
                raise ValueError(f"selected run is incomplete: {directory}")
            continue  # failed attempts are listed by the run manifest, not compared
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("status") != "complete":
            raise ValueError(f"run is not complete: {directory}")
        seed = int(metrics["seed"])
        if seed not in seeds:
            continue
        if (method, seed) in runs:
            raise ValueError(f"duplicate {method} seed {seed}; select runs with --runs")
        provenance.add(tuple(metrics.get(field) for field in (
            "dataset_sha256", "source_sha256", "evaluation_split", "backbone", "epochs")))
        runs[(method, seed)] = (directory, config, metrics)
    missing = [(method, seed) for method in methods for seed in seeds if (method, seed) not in runs]
    if missing:
        raise ValueError(f"missing completed runs: {missing}")
    if len(provenance) != 1:
        raise ValueError("runs differ in data, source revision, evaluation stage, backbone or epochs: "
                         f"{sorted(provenance, key=str)}")
    return runs


def endpoint_keys(config: dict, descriptor: str) -> dict[str, list[str]]:
    evaluation = config["evaluation"]
    families = list(evaluation["corruption_families"])
    severities = [float(value) for value in evaluation["severities"] if float(value) > 0]
    keys = {"clean": [f"{descriptor}_clean_gallery_{families[0]}_0"],
            "robust": [f"{descriptor}_clean_gallery_{f}_{s:g}" for f in families for s in severities]}
    if "occluded" in evaluation.get("gallery_protocols", ["clean"]):
        keys["occluded"] = [f"{descriptor}_occluded_gallery_{f}_{s:g}" for f in families for s in severities]
    for family in families:
        keys[f"family:{family}"] = [f"{descriptor}_clean_gallery_{family}_{s:g}" for s in severities]
    return keys


def identity_blocks(runs, method, seeds, keys, prefix=""):
    """[seed, identity] identity-level means of the per-probe endpoint value."""
    blocks, reference = [], None
    for seed in seeds:
        with np.load(runs[(method, seed)][0] / "probe_outcomes.npz", allow_pickle=False) as archive:
            order = (archive["probe_indices"], archive["probe_labels"])
            if reference is None:
                reference = tuple(value.copy() for value in order)
            elif not all(np.array_equal(a, b) for a, b in zip(reference, order)):
                raise ValueError(f"probe ordering differs for {method} seed {seed}")
            missing = [prefix + key for key in keys if prefix + key not in archive.files]
            if missing:
                return None, reference
            value = np.mean(np.stack([archive[prefix + key].astype(np.float64) for key in keys]), axis=0)
        labels = reference[1]
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite per-probe values for {method} seed {seed}")
        blocks.append([value[labels == identity].mean() for identity in np.unique(labels)])
    return np.asarray(blocks), reference


def interval(values, samples, rng):
    draws = crossed_draws(values, samples, rng)
    return {"mean": float(values.mean()), "ci95_low": float(np.quantile(draws, .025)),
            "ci95_high": float(np.quantile(draws, .975)), "probability_positive": float(np.mean(draws > 0))}


def official_rows(runs, methods, seeds, descriptors):
    rows = []
    for method in methods:
        for descriptor in descriptors:
            cells = {}
            for seed in seeds:
                for row in runs[(method, seed)][2]["results"]:
                    if row["protocol"] == "single_view_gallery" and row["descriptor"] == descriptor:
                        cells.setdefault(row["corruption"].split(":")[-1], []).append(float(row["rank1"]))
            for condition, values in sorted(cells.items()):
                if len(values) == len(seeds):
                    rows.append({"method": method, "descriptor": descriptor, "condition": condition,
                                 "rank1_mean": statistics.fmean(values),
                                 "rank1_sd": statistics.stdev(values) if len(values) > 1 else None})
    return rows


def write_csv(path: Path, rows: list[dict]):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def compare(root: Path, output: Path, dataset: str, methods: list[str], seeds: list[int],
            reference: str, samples: int, bootstrap_seed: int, run_names=None):
    if reference not in methods or len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError("methods and seeds must be unique, and the reference must be a compared method")
    runs = discover(root, dataset, methods, seeds, run_names)
    config = runs[(reference, seeds[0])][1]
    descriptors = list(config["evaluation"]["descriptors"])
    for (method, seed), (_, current, _) in runs.items():
        if list(current["evaluation"]["descriptors"]) != descriptors:
            raise ValueError(f"descriptor set differs for {method} seed {seed}")
    rng = np.random.default_rng(bootstrap_seed)
    summary, contrasts, cache, unavailable = [], [], {}, set()
    probe_reference = None
    for method in methods:
        for descriptor in descriptors:
            for endpoint, keys in endpoint_keys(config, descriptor).items():
                for metric, prefix in (("rank1", ""), ("map", AP_PREFIX)):
                    blocks, order = identity_blocks(runs, method, seeds, keys, prefix)
                    if probe_reference is None:
                        probe_reference = order
                    elif not all(np.array_equal(a, b) for a, b in zip(probe_reference, order)):
                        raise ValueError(f"probe ordering differs between methods ({method})")
                    if blocks is None:
                        unavailable.add(metric)
                        continue
                    cache[(method, descriptor, endpoint, metric)] = blocks
                    summary.append({"method": method, "descriptor": descriptor, "endpoint": endpoint,
                                    "metric": metric, **interval(blocks, samples, rng),
                                    **{f"seed{seed}": float(value) for seed, value in zip(seeds, blocks.mean(1))}})
    for (method, descriptor, endpoint, metric), blocks in cache.items():
        if method != reference:
            baseline = cache[(reference, descriptor, endpoint, metric)]
            contrasts.append({"type": "method", "comparison": f"{method}_minus_{reference}", "method": method,
                              "descriptor": descriptor, "endpoint": endpoint, "metric": metric,
                              **interval(blocks - baseline, samples, rng)})
    for method in methods:
        for candidate, baseline in DESCRIPTOR_CONTRASTS:
            for (current, descriptor, endpoint, metric), blocks in cache.items():
                if current == method and descriptor == candidate and (method, baseline, endpoint, metric) in cache:
                    contrasts.append({"type": "descriptor", "comparison": f"{candidate}_minus_{baseline}",
                                      "method": method, "descriptor": candidate, "endpoint": endpoint,
                                      "metric": metric, **interval(
                                          blocks - cache[(method, baseline, endpoint, metric)], samples, rng)})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison_summary.csv", summary)
    write_csv(output / "comparison_contrasts.csv", contrasts)
    official = official_rows(runs, methods, seeds, descriptors)
    write_csv(output / "official_single_view.csv", official)
    first = runs[(reference, seeds[0])][2]
    result = {
        "dataset": dataset, "evaluation_split": first["evaluation_split"], "methods": methods,
        "reference": reference, "seeds": seeds, "descriptors": descriptors,
        "probes": int(len(probe_reference[0])), "identities": int(np.unique(probe_reference[1]).size),
        "dataset_sha256": first["dataset_sha256"], "source_sha256": first["source_sha256"],
        "epochs": first["epochs"], "runs": {f"{m}:{s}": runs[(m, s)][0].name for m, s in sorted(runs)},
        "unavailable_metrics": sorted(unavailable - {metric for *_, metric in cache}),
        "bootstrap": {"samples": samples, "seed": bootstrap_seed, "design": "crossed seed x identity"},
        "note": ("Contrasts pair methods by seed index and identity. Intervals are pointwise and "
                 "unadjusted for multiple comparisons."),
    }
    if "map" in result["unavailable_metrics"]:
        result["note"] += " These runs predate per-probe AP storage, so mAP has no interval."
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", required=True, help="comma-separated method names")
    parser.add_argument("--reference", default="metric_baseline")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--runs", nargs="+", help="run directory names under RESULTS")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261008)
    args = parser.parse_args()
    result = compare(args.results, args.output or args.results / "comparison",
                     args.dataset, args.methods.split(","), [int(v) for v in args.seeds.split(",")],
                     args.reference, args.bootstrap_samples, args.bootstrap_seed, args.runs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
