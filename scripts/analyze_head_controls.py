#!/usr/bin/env python3
"""Compare P1 frozen-head runs with the fixed P0 raw descriptor."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


MODES = ("distill", "triplet_variance")
FAMILIES = ("random_joint", "dynamic_joint", "body_part", "temporal")
SEVERITIES = ("0.1", "0.3", "0.5", "0.7")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=Path("results_head_controls"))
    parser.add_argument("--p0-results", type=Path, default=Path("results_descriptor_controls"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    return parser.parse_args()


def discover_p1(root, seeds):
    runs = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        required = [directory / name for name in ("config.yaml", "history.json", "metrics.json", "probe_outcomes.npz")]
        if not any(path.exists() for path in required):
            continue
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise SystemExit(f"{directory} is incomplete: missing {', '.join(missing)}")
        config = yaml.safe_load((directory / "config.yaml").read_text())
        if config.get("experiment_id") != "frozen_head_p1":
            continue
        key = (config["head_training"]["mode"], int(config["seed"]))
        if key in runs:
            raise SystemExit(f"duplicate P1 run for {key}")
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics.get("status") != "complete" or metrics.get("evaluation_split") != "validation":
            raise SystemExit(f"{directory} is not a complete validation run")
        runs[key] = directory
    expected = {(mode, seed) for mode in MODES for seed in seeds}
    if set(runs) != expected:
        raise SystemExit(
            f"P1 design mismatch; missing={sorted(expected.difference(runs))}, "
            f"extra={sorted(set(runs).difference(expected))}"
        )
    return runs


def discover_p0(root, seeds):
    runs = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = directory / "config.yaml"
        if not config_path.exists() or not (directory / "probe_outcomes.npz").exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        if config.get("experiment_id") == "descriptor_controls_p0":
            runs[int(config["seed"])] = directory
    if set(runs) != set(seeds):
        raise SystemExit(f"P0 references do not contain exactly seeds {seeds}")
    return runs


def metric_endpoint(rows, descriptor, endpoint):
    selected = [row for row in rows if row["descriptor"] == descriptor]
    if endpoint == "clean":
        values = [row["rank1"] for row in selected if row["protocol"] == "clean_gallery"
                  and row["corruption"] == "body_part" and float(row["severity"]) == 0]
    elif endpoint == "robust":
        values = [row["rank1"] for row in selected if row["protocol"] == "clean_gallery"
                  and row["corruption"] in FAMILIES and float(row["severity"]) > 0]
    elif endpoint == "severity_05":
        values = [row["rank1"] for row in selected if row["protocol"] == "clean_gallery"
                  and row["corruption"] in FAMILIES and float(row["severity"]) == .5]
    elif endpoint == "occluded":
        values = [row["rank1"] for row in selected if row["protocol"] == "occluded_gallery"
                  and row["corruption"] in FAMILIES and float(row["severity"]) > 0]
    else:
        raise ValueError(endpoint)
    if not values:
        raise SystemExit(f"missing {endpoint} for {descriptor}")
    return float(np.mean(values))


def raw_metric_endpoint(rows, endpoint):
    selected = [
        row for row in rows
        if row["descriptor"] == "raw_reliable" and float(row["fusion_weight"]) == .5
    ]
    return metric_endpoint(selected, "raw_reliable", endpoint)


def p1_key(endpoint):
    if endpoint == "clean":
        return ["reliable_parts_clean_gallery_body_part_0"]
    if endpoint == "robust":
        return [f"reliable_parts_clean_gallery_{family}_{severity}"
                for family in FAMILIES for severity in SEVERITIES]
    if endpoint == "severity_05":
        return [f"reliable_parts_clean_gallery_{family}_0.5" for family in FAMILIES]
    if endpoint == "occluded":
        return [f"reliable_parts_occluded_gallery_{family}_{severity}"
                for family in FAMILIES for severity in SEVERITIES]
    raise ValueError(endpoint)


def p0_key(endpoint):
    return [f"raw_reliable_w0.5_{key[len('reliable_parts_'):]}" for key in p1_key(endpoint)]


def averaged(archive, keys):
    missing = set(keys).difference(archive.files)
    if missing:
        raise SystemExit(f"probe archive is missing keys: {sorted(missing)}")
    return np.nanmean(np.stack([archive[key] for key in keys]), axis=0)


def blocks(p1_runs, p0_runs, seeds, mode, endpoint):
    values = []
    for seed in seeds:
        with np.load(p1_runs[(mode, seed)] / "probe_outcomes.npz") as head, np.load(
            p0_runs[seed] / "probe_outcomes.npz"
        ) as raw:
            if not np.array_equal(head["probe_indices"], raw["probe_indices"]):
                raise SystemExit(f"probe ordering mismatch for {mode}, seed {seed}")
            labels = head["probe_labels"]
            if not np.array_equal(labels, raw["probe_labels"]):
                raise SystemExit(f"probe label mismatch for {mode}, seed {seed}")
            before, after = averaged(raw, p0_key(endpoint)), averaged(head, p1_key(endpoint))
            values.append([
                float(np.nanmean(after[labels == identity] - before[labels == identity]))
                for identity in np.unique(labels)
            ])
    return np.asarray(values)


def bootstrap(values, samples, rng):
    seed_count, identity_count = values.shape
    seed_draws = rng.integers(0, seed_count, size=(samples, seed_count))
    identity_draws = rng.integers(0, identity_count, size=(samples, seed_count, identity_count))
    draws = np.take_along_axis(values[seed_draws], identity_draws, axis=2).mean(axis=(1, 2))
    return {
        "difference": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [.025, .975])],
        "probability_positive": float(np.mean(draws > 0)),
    }


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    p1_runs, p0_runs = discover_p1(args.results, seeds), discover_p0(args.p0_results, seeds)
    endpoints = ("clean", "robust", "severity_05", "occluded")
    p0_metrics = {
        seed: json.loads((p0_runs[seed] / "metrics.json").read_text())["results"] for seed in seeds
    }
    raw = {
        endpoint: float(np.mean([raw_metric_endpoint(p0_metrics[seed], endpoint) for seed in seeds]))
        for endpoint in endpoints
    }
    rng = np.random.default_rng(args.bootstrap_seed)
    analysis, summary_rows = {}, []
    global_clean = []
    for seed in seeds:
        rows = json.loads((p1_runs[(MODES[0], seed)] / "metrics.json").read_text())["results"]
        global_clean.append(metric_endpoint(rows, "global", "clean"))
    global_clean = float(np.mean(global_clean))
    for mode in MODES:
        metrics = [json.loads((p1_runs[(mode, seed)] / "metrics.json").read_text()) for seed in seeds]
        absolute = {
            endpoint: float(np.mean([
                metric_endpoint(item["results"], "reliable_parts", endpoint) for item in metrics
            ])) for endpoint in endpoints
        }
        paired = {
            endpoint: bootstrap(
                blocks(p1_runs, p0_runs, seeds, mode, endpoint), args.bootstrap_samples, rng
            ) for endpoint in endpoints
        }
        variances = [float(item["final_part_variance"]) for item in metrics]
        safe = absolute["clean"] >= global_clean - .005 and min(variances) >= 1e-4
        preserves_raw_clean = absolute["clean"] >= raw["clean"] - .005
        replaces = safe and preserves_raw_clean and paired["robust"]["ci95"][0] >= 0
        analysis[mode] = {
            "absolute": absolute, "difference_from_raw": paired,
            "selected_part_variance": variances,
            "clean_safety_pass": safe, "preserves_raw_clean": preserves_raw_clean,
            "replaces_raw": replaces,
        }
        summary_rows.append({
            "mode": mode, **absolute,
            "clean_difference_from_raw": paired["clean"]["difference"],
            "robust_difference_from_raw": paired["robust"]["difference"],
            "robust_ci95_low": paired["robust"]["ci95"][0],
            "robust_ci95_high": paired["robust"]["ci95"][1],
            "minimum_selected_part_variance": min(variances),
            "replaces_raw": replaces,
        })
    winner = next((mode for mode in MODES if analysis[mode]["replaces_raw"]), None)
    payload = {
        "design": {"modes": list(MODES), "seeds": seeds, "bootstrap_samples": args.bootstrap_samples},
        "raw_reference": raw, "global_clean_reference": global_clean,
        "heads": analysis,
        "recommendation": f"use_{winner}_head" if winner else "retain_raw_descriptor",
    }
    output = args.output or args.results
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    print(json.dumps({"runs": len(p1_runs), "recommendation": payload["recommendation"], "heads": analysis}, indent=2))


if __name__ == "__main__":
    main()
