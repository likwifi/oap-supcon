#!/usr/bin/env python3
"""Audit a possibly incomplete V3 screen and compare matched seeds/identities.

Confidence intervals resample training seeds and validation identities as crossed
clusters: the same identity draw is shared by every sampled training seed. They
are exploratory, pointwise 95% intervals, without a multiple-comparison correction.
The paired endpoints use pooled galleries; official single-view NM/BG/CL scores
are reported separately and are not bootstrapped from pooled-gallery outcomes.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oap_supcon.provenance import configuration_fingerprint, source_fingerprint

ARMS = ("control", "angles", "late_fusion", "adaptive", "parts", "pooling", "full")
SEEDS = (11, 22, 33)
DESCRIPTORS = ("global", "raw_reliable", "raw_uniform")
FAMILIES = ("random_joint", "dynamic_joint", "body_part", "temporal")
SEVERITIES = (.1, .3, .5, .7)
ENDPOINTS = ("clean", "robust", "severity_05", "occluded", *FAMILIES)


def outcome_keys(descriptor, endpoint):
    if endpoint == "clean":
        cells = [("clean", "body_part", 0)]
    else:
        protocol = "occluded" if endpoint == "occluded" else "clean"
        families = (endpoint,) if endpoint in FAMILIES else FAMILIES
        severities = (.5,) if endpoint == "severity_05" else SEVERITIES
        cells = [(protocol, family, severity) for family in families for severity in severities]
    return [f"{descriptor}_{protocol}_gallery_{family}_{severity:g}"
            for protocol, family, severity in cells]


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def interval(values, samples, rng):
    seed_count, identity_count = values.shape
    seed_draws = rng.integers(seed_count, size=(samples, seed_count))
    identity_draws = rng.integers(identity_count, size=(samples, identity_count))
    draws = values[seed_draws[:, :, None], identity_draws[:, None, :]].mean(axis=(1, 2))
    return {"difference": float(values.mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "positive_fraction": float(np.mean(draws > 0))}


def analyze(results, output, samples, bootstrap_seed):
    runs, manifest, sources, datasets, environments = {}, [], set(), set(), set()
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    reference_indices = reference_labels = reference_train = reference_held = reference_cells = None
    fingerprints = {}
    for directory in sorted(path for path in results.iterdir() if path.is_dir()):
        config_path = directory / "config.yaml"
        if not config_path.exists():
            continue
        cfg = yaml.safe_load(config_path.read_text())
        experiment = cfg.get("experiment_id", "")
        if not experiment.startswith("benchmark_v3_"):
            continue
        arm, seed = experiment.removeprefix("benchmark_v3_"), int(cfg["seed"])
        key = (arm, seed)
        if key not in expected:
            raise ValueError(f"unexpected arm/seed: {key}")
        if configuration_fingerprint(cfg) != cfg["config_fingerprint"]:
            raise ValueError(f"configuration fingerprint mismatch: {directory}")
        if arm in fingerprints and fingerprints[arm] != cfg["config_fingerprint"]:
            raise ValueError(f"recipe differs across seeds for {arm}")
        fingerprints[arm] = cfg["config_fingerprint"]
        sources.add(cfg["source_sha256"])
        datasets.add(cfg["dataset_sha256"])
        train, held = cfg["train_identities"], cfg["validation"]["held_out_identities"]
        if reference_train is None:
            reference_train, reference_held = train, held
        if train != reference_train or held != reference_held or set(train).intersection(held):
            raise ValueError(f"training/validation identity mismatch: {directory}")
        if not set(train + held).issubset(set(range(1, 75))) or len(train) != 62 or len(held) != 12:
            raise ValueError("the screen must use 62/12 identities from CASIA-B training")
        if cfg["epochs"] != 80 or cfg["evaluation"]["split"] != "validation":
            raise ValueError("expected an 80-epoch validation-only screen")
        if set(cfg["evaluation"]["descriptors"]) != set(DESCRIPTORS) or cfg["evaluation"]["part_weight"] != .5:
            raise ValueError("the frozen descriptor comparison differs")
        metric_path = directory / "metrics.json"
        history_path = directory / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        failure_path = directory / "failure.json"
        failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
        task = ARMS.index(arm) * len(SEEDS) + SEEDS.index(seed)
        entry = {"task_id": task, "arm": arm, "seed": seed, "run_id": directory.name,
                 "status": "complete" if metric_path.exists() else "failed" if failure else "incomplete",
                 "saved_epoch": history[-1]["epoch"] if history else 0,
                 "has_checkpoint": (directory / "checkpoint.pt").exists(),
                 "failure_message": failure.get("message", "")}
        manifest.append(entry)
        if not metric_path.exists():
            continue
        if key in runs:
            raise ValueError(f"duplicate completed run for {key}")
        metrics = json.loads(metric_path.read_text())
        if metrics["status"] != "complete" or metrics["evaluation_split"] != "validation":
            raise ValueError(f"invalid completed run: {directory}")
        for field in ("source_sha256", "dataset_sha256", "config_fingerprint", "seed", "epochs", "experiment_id"):
            if cfg[field] != metrics[field]:
                raise ValueError(f"config/metrics disagree on {field}: {directory}")
        if [row["epoch"] for row in history] != list(range(1, 81)):
            raise ValueError(f"history does not contain 80 completed epochs: {directory}")
        selection = {row["epoch"]: row["validation"]["score"] for row in history if "validation" in row}
        if metrics["selected_epoch"] != max(selection, key=selection.get):
            raise ValueError(f"selected checkpoint is not the first best validation score: {directory}")
        environments.add((metrics["environment"]["python"], metrics["environment"]["torch"]))
        cells = {(r["descriptor"], r["protocol"], r["corruption"], float(r["severity"]), r.get("probe_view", ""))
                 for r in metrics["results"]}
        if len(cells) != len(metrics["results"]):
            raise ValueError(f"duplicate metric cells: {directory}")
        if reference_cells is None:
            reference_cells = cells
        if cells != reference_cells:
            raise ValueError(f"metric cells differ: {directory}")
        with np.load(directory / "probe_outcomes.npz", allow_pickle=False) as archive:
            labels, indices = archive["probe_labels"], archive["probe_indices"]
            if reference_indices is None:
                reference_indices, reference_labels = indices.copy(), labels.copy()
            if not np.array_equal(indices, reference_indices) or not np.array_equal(labels, reference_labels):
                raise ValueError(f"probe order/labels differ: {directory}")
            if set(np.unique(labels)) != set(held):
                raise ValueError(f"probe identities differ from validation identities: {directory}")
            for row in metrics["results"]:
                if row["corruption"] not in FAMILIES:
                    continue
                outcome = archive[f"{row['descriptor']}_{row['protocol']}_{row['corruption']}_{row['severity']:g}"]
                if not np.isfinite(outcome).all() or abs(float(outcome.mean()) - row["rank1"]) > 1e-6:
                    raise ValueError(f"probe outcomes disagree with Rank-1: {directory}")
            endpoints = {(d, e): np.stack([archive[k] for k in outcome_keys(d, e)]).mean(axis=0)
                         for d in DESCRIPTORS for e in ENDPOINTS}
        runs[key] = {"metrics": metrics, "history": history, "endpoints": endpoints, "labels": labels}
    if not runs or len(sources) != 1 or len(datasets) != 1 or len(environments) != 1:
        raise ValueError("the screen needs completed runs with matching source/data/environment")
    rows, seed_rows, comparisons, rng = [], [], [], np.random.default_rng(bootstrap_seed)
    for (arm, seed), run in sorted(runs.items()):
        metrics = run["metrics"]
        for descriptor in DESCRIPTORS:
            conditions = {r["corruption"].split(":")[-1]: r["rank1"] for r in metrics["results"]
                          if r["descriptor"] == descriptor and r["protocol"] == "single_view_gallery"}
            seed_rows.append({"arm": arm, "seed": seed, "descriptor": descriptor,
                              **{e: float(run["endpoints"][(descriptor, e)].mean()) for e in ENDPOINTS},
                              **{f"official_{c}": conditions[c] for c in ("NM", "BG", "CL")},
                              "parameters": metrics["parameters"], "selected_epoch": metrics["selected_epoch"],
                              "train_hours": metrics["train_hours"], "gpu": metrics["environment"]["gpu"]})
    for arm in ARMS:
        seeds = [s for s in SEEDS if (arm, s) in runs]
        for descriptor in DESCRIPTORS:
            selected = [r for r in seed_rows if r["arm"] == arm and r["descriptor"] == descriptor]
            if not selected:
                continue
            row = {"arm": arm, "descriptor": descriptor, "seeds": len(seeds), "complete_arm": len(seeds) == 3}
            for endpoint in (*ENDPOINTS, "official_NM", "official_BG", "official_CL", "train_hours"):
                values = [r[endpoint] for r in selected]
                row[endpoint] = statistics.fmean(values)
                row[f"{endpoint}_std"] = statistics.stdev(values) if len(values) > 1 else None
            row["parameters"] = selected[0]["parameters"]
            rows.append(row)
    identities = np.unique(reference_labels)
    pairs = [(arm, descriptor, "control", descriptor, "architecture")
             for arm in ARMS if arm != "control" for descriptor in DESCRIPTORS]
    pairs += [(arm, candidate, arm, baseline, "descriptor") for arm in ARMS
              for candidate, baseline in (("raw_reliable", "global"), ("raw_uniform", "global"),
                                          ("raw_reliable", "raw_uniform"))]
    for arm, descriptor, base_arm, base_descriptor, kind in pairs:
        seeds = [s for s in SEEDS if (arm, s) in runs and (base_arm, s) in runs]
        if not seeds:
            continue
        for endpoint in ENDPOINTS:
            deltas = [runs[(arm, s)]["endpoints"][(descriptor, endpoint)] -
                      runs[(base_arm, s)]["endpoints"][(base_descriptor, endpoint)] for s in seeds]
            blocks = np.array([[float(delta[reference_labels == identity].mean()) for identity in identities]
                               for delta in deltas])
            comparisons.append({"kind": kind, "candidate_arm": arm, "candidate_descriptor": descriptor,
                                "baseline_arm": base_arm, "baseline_descriptor": base_descriptor,
                                "endpoint": endpoint, "seeds": seeds, "seed_count": len(seeds),
                                "seed_differences": [float(delta.mean()) for delta in deltas],
                                **interval(blocks, samples, rng)})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "run_manifest.csv", manifest)
    write_csv(output / "architecture_summary.csv", rows)
    write_csv(output / "per_seed.csv", seed_rows)
    write_csv(output / "paired_comparisons.csv", comparisons)
    payload = {"completed_runs": len(runs), "expected_runs": len(expected),
               "missing_tasks": [ARMS.index(a) * 3 + SEEDS.index(s) for a, s in sorted(expected.difference(runs),
                                                                                  key=lambda k: (ARMS.index(k[0]), k[1]))],
               "source_sha256": next(iter(sources)), "source_matches_local": next(iter(sources)) == source_fingerprint(ROOT),
               "dataset_sha256": next(iter(datasets)), "probe_count": len(reference_labels),
               "validation_identities": identities.tolist(), "training_identities": reference_train,
               "environment": [list(e) for e in environments], "bootstrap_samples": samples,
               "bootstrap_seed": bootstrap_seed, "bootstrap_method": "crossed paired seed/identity resampling",
               "interval_limitations": "Pointwise exploratory 95% intervals; no multiplicity adjustment; three seeds and 12 identities.",
               "summary": rows, "comparisons": comparisons, "manifest": manifest}
    (output / "analysis.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results_v3_dev"))
    parser.add_argument("--output", type=Path, default=Path("audits/benchmark_v3_completed"))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261007)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap samples must be positive")
    result = analyze(args.results, args.output, args.bootstrap_samples, args.bootstrap_seed)
    print(json.dumps({key: result[key] for key in ("completed_runs", "expected_runs", "missing_tasks", "source_matches_local", "probe_count")}))


if __name__ == "__main__":
    main()
