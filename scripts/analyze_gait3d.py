#!/usr/bin/env python3
"""Summarize matched Gait3D descriptor runs and paired seed/identity intervals."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oap_supcon.reporting import aggregate_results  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_stats import crossed_draws  # noqa: E402

DESCRIPTORS = ("global", "raw_reliable", "raw_uniform")


def discover_runs(root: Path, seeds: list[int], run_names: list[str] | None = None):
    """`run_names` selects matched runs from a root that also holds failed or duplicate attempts."""
    runs = {}
    signatures = set()
    paths = (sorted(root.glob("*/config.yaml")) if run_names is None
             else [root / name / "config.yaml" for name in run_names])
    for path in paths:
        if not path.exists():
            raise ValueError(f"selected run has no config.yaml: {path.parent}")
        config = yaml.safe_load(path.read_text())
        if config.get("dataset", {}).get("name") != "gait3d" or config.get("method_name") != "metric_baseline":
            continue
        directory = path.parent
        if not all((directory / name).exists() for name in ("metrics.json", "probe_outcomes.npz")):
            raise ValueError(f"incomplete run: {directory}")
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics.get("status") != "complete":
            raise ValueError(f"incomplete run: {directory}")
        seed = int(metrics["seed"])
        if seed in runs:
            raise ValueError(f"duplicate seed {seed}; use a directory containing one evaluation stage")
        signature = tuple(metrics.get(field) for field in (
            "dataset_sha256", "source_sha256", "config_fingerprint", "evaluation_split", "backbone"))
        if any(value is None for value in signature):
            raise ValueError(f"run is missing scientific provenance: {directory}")
        signatures.add(signature)
        if set(config["evaluation"]["descriptors"]) != set(DESCRIPTORS) or config["evaluation"]["part_weight"] != .5:
            raise ValueError("comparison requires the three frozen descriptors at weight 0.5")
        runs[seed] = (directory, metrics)
    if set(runs) != set(seeds):
        raise ValueError(f"seed mismatch: expected {seeds}, found {sorted(runs)}")
    if len(signatures) != 1:
        raise ValueError("runs have different data, source, recipe, backbone or evaluation stage")
    return runs


def endpoint_rows(rows, descriptor, endpoint):
    selected = [r for r in rows if r["descriptor"] == descriptor
                and r["corruption"] in {"random_joint", "dynamic_joint", "body_part", "temporal"}]
    if endpoint == "clean":
        selected = [r for r in selected if r["protocol"] == "clean_gallery" and float(r["severity"]) == 0]
        return selected[:1]
    protocol = "occluded_gallery" if endpoint == "occluded" else "clean_gallery"
    return [r for r in selected if r["protocol"] == protocol and float(r["severity"]) > 0
            and (endpoint != "severity_05" or float(r["severity"]) == .5)]


def paired_interval(blocks, samples, rng):
    """Crossed seed x identity interval; every seed scores the same identities."""
    blocks = np.asarray(blocks, dtype=np.float64)
    draws = crossed_draws(blocks, samples, rng)
    return {"difference": float(blocks.mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist()}


def analyze(root: Path, output: Path, seeds: list[int], samples: int, bootstrap_seed: int):
    if samples < 1 or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("bootstrap samples and unique seeds must be positive")
    runs = discover_runs(root, seeds)
    output.mkdir(parents=True, exist_ok=True)
    aggregate_results(root, duplicates="error", output_root=output)
    summary, paired = [], {}
    rng = np.random.default_rng(bootstrap_seed)
    reference_order = reference_labels = None
    for endpoint in ("clean", "robust", "severity_05", "occluded"):
        selections = {seed: {descriptor: endpoint_rows(metrics["results"], descriptor, endpoint)
                             for descriptor in DESCRIPTORS} for seed, (_, metrics) in runs.items()}
        if not any(selections[seed]["global"] for seed in seeds):
            continue
        if not all(selections[seed][descriptor] for seed in seeds for descriptor in DESCRIPTORS):
            raise ValueError(f"missing descriptor/seed cells for {endpoint}")
        cells = {(r["protocol"], r["corruption"], float(r["severity"]))
                 for r in selections[seeds[0]]["global"]}
        for seed in seeds:
            for descriptor in DESCRIPTORS:
                actual = {(r["protocol"], r["corruption"], float(r["severity"]))
                          for r in selections[seed][descriptor]}
                if actual != cells:
                    raise ValueError(f"corruption cells differ for {endpoint}, {descriptor}, seed {seed}")
        for descriptor in DESCRIPTORS:
            row = {"endpoint": endpoint, "descriptor": descriptor, "seeds": len(seeds)}
            for metric in ("rank1", "rank5", "rank10", "map", "minp"):
                values = [float(np.mean([r[metric] for r in selections[seed][descriptor]])) for seed in seeds]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(row)
        for candidate, baseline in (("raw_reliable", "global"), ("raw_uniform", "global"),
                                    ("raw_reliable", "raw_uniform")):
            blocks = []
            for seed in seeds:
                with np.load(runs[seed][0] / "probe_outcomes.npz", allow_pickle=False) as archive:
                    labels, order = archive["probe_labels"], archive["probe_indices"]
                    if reference_order is None:
                        reference_order, reference_labels = order.copy(), labels.copy()
                    if not np.array_equal(order, reference_order) or not np.array_equal(labels, reference_labels):
                        raise ValueError("probe ordering or labels differ across seeds")
                    outcomes = {}
                    for descriptor in (candidate, baseline):
                        keys = [f"{descriptor}_{protocol}_{family}_{severity:g}" for protocol, family, severity in sorted(cells)]
                        outcomes[descriptor] = np.mean(np.stack([archive[key] for key in keys]), axis=0)
                    differences = outcomes[candidate] - outcomes[baseline]
                    if not np.isfinite(differences).all():
                        raise ValueError("paired outcomes contain nonfinite values")
                    blocks.append(np.asarray([differences[labels == identity].mean() for identity in np.unique(labels)]))
            paired[f"{candidate}_minus_{baseline}:{endpoint}"] = paired_interval(blocks, samples, rng)
    if not summary:
        raise ValueError("runs contain no descriptor comparison endpoints")
    with (output / "descriptor_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    first = runs[seeds[0]][1]
    result = {"dataset": "gait3d", "evaluation_split": first["evaluation_split"], "seeds": seeds,
              "source_coverage": first.get("source_coverage"), "dataset_sha256": first["dataset_sha256"],
              "bootstrap_samples": samples, "bootstrap_seed": bootstrap_seed, "paired": paired}
    (output / "paired_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261005)
    args = parser.parse_args()
    result = analyze(args.results, args.output or args.results / "comparison",
                     [int(v) for v in args.seeds.split(",")], args.bootstrap_samples, args.bootstrap_seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
