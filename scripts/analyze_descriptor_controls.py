#!/usr/bin/env python3
"""Validate and summarize the three checkpoint-only P0 control runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


FAMILIES = ["random_joint", "dynamic_joint", "body_part", "temporal"]
SEVERITIES = [0.1, 0.3, 0.5, 0.7]
PRIMARY_WEIGHT = 0.5
DETERMINISTIC = ["raw_reliable", "raw_uniform"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=Path("results_descriptor_controls"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261002)
    return parser.parse_args()


def discover(root: Path, expected_seeds: list[int]):
    runs = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        required = [directory / name for name in ("config.yaml", "metrics.json", "probe_outcomes.npz")]
        if not any(path.exists() for path in required):
            continue
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise SystemExit(f"{directory} is incomplete: missing {', '.join(missing)}")
        config = yaml.safe_load((directory / "config.yaml").read_text())
        if config.get("experiment_id") != "descriptor_controls_p0":
            continue
        seed = int(config["seed"])
        if seed in runs:
            raise SystemExit(f"duplicate P0 run for seed {seed}")
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics.get("status") != "complete":
            raise SystemExit(f"{directory} is not complete")
        if metrics.get("evaluation_split") != "validation":
            raise SystemExit(f"{directory} accessed a non-validation split")
        runs[seed] = directory
    missing = sorted(set(expected_seeds).difference(runs))
    extra = sorted(set(runs).difference(expected_seeds))
    if missing or extra:
        raise SystemExit(f"P0 design mismatch; missing={missing}, extra={extra}")
    configs = [yaml.safe_load((runs[seed] / "config.yaml").read_text()) for seed in expected_seeds]
    for field in ("random_head_seeds", "fusion_weights", "held_out_identities"):
        if any(config[field] != configs[0][field] for config in configs[1:]):
            raise SystemExit(f"P0 runs disagree on {field}")
    if len(configs[0]["random_head_seeds"]) < 10:
        raise SystemExit("P0 design contains fewer than 10 random-head seeds")
    return runs, configs[0]


def result_rows(runs, seeds):
    return {
        seed: json.loads((runs[seed] / "metrics.json").read_text())["results"]
        for seed in seeds
    }


def aggregate_rows(by_seed, seeds):
    grouped = defaultdict(list)
    for seed in seeds:
        for row in by_seed[seed]:
            key = (
                row["descriptor"], float(row.get("fusion_weight", 0)), row["protocol"],
                row["corruption"], float(row["severity"]),
            )
            grouped[key].append(float(row["rank1"]))
    output = []
    for key, values in sorted(grouped.items()):
        if len(values) != len(seeds):
            raise SystemExit(f"aggregate cell {key} has {len(values)} rather than {len(seeds)} seeds")
        output.append({
            "descriptor": key[0], "fusion_weight": key[1], "protocol": key[2],
            "corruption": key[3], "severity": key[4],
            "rank1_mean": float(np.mean(values)), "rank1_std": float(np.std(values, ddof=1)),
            "rank1_count": len(values),
        })
    return output


def endpoint(rows, descriptor, weight, name):
    selected = [
        row for row in rows
        if row["descriptor"] == descriptor and float(row.get("fusion_weight", 0)) == weight
    ]
    if name == "clean":
        values = [row["rank1"] for row in selected if row["protocol"] == "clean_gallery"
                  and row["corruption"] == "body_part" and float(row["severity"]) == 0]
    elif name == "robust":
        values = [row["rank1"] for row in selected if row["protocol"] == "clean_gallery"
                  and row["corruption"] in FAMILIES and float(row["severity"]) > 0]
    elif name == "severity_05":
        values = [row["rank1"] for row in selected if row["protocol"] == "clean_gallery"
                  and row["corruption"] in FAMILIES and float(row["severity"]) == .5]
    elif name == "occluded":
        values = [row["rank1"] for row in selected if row["protocol"] == "occluded_gallery"
                  and row["corruption"] in FAMILIES and float(row["severity"]) > 0]
    elif name.startswith("family:"):
        family = name.split(":", 1)[1]
        values = [row["rank1"] for row in selected if row["protocol"] == "clean_gallery"
                  and row["corruption"] == family and float(row["severity"]) > 0]
    else:
        raise ValueError(name)
    if not values:
        raise SystemExit(f"missing endpoint {name} for {descriptor} at weight {weight}")
    return float(np.mean(values))


def aggregate_endpoints(by_seed, seeds, descriptor, weight):
    names = ["clean", "robust", "severity_05", "occluded"] + [f"family:{f}" for f in FAMILIES]
    return {
        name: float(np.mean([endpoint(by_seed[seed], descriptor, weight, name) for seed in seeds]))
        for name in names
    }


def outcome_key(descriptor, weight, protocol, corruption, severity):
    import re
    value = f"{descriptor}_w{weight:g}_{protocol}_{corruption}_{severity:g}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def endpoint_keys(descriptor, weight, name):
    if name == "clean":
        return [outcome_key(descriptor, weight, "clean_gallery", "body_part", 0)]
    if name == "robust":
        return [outcome_key(descriptor, weight, "clean_gallery", family, severity)
                for family in FAMILIES for severity in SEVERITIES]
    if name == "severity_05":
        return [outcome_key(descriptor, weight, "clean_gallery", family, .5) for family in FAMILIES]
    if name == "occluded":
        return [outcome_key(descriptor, weight, "occluded_gallery", family, severity)
                for family in FAMILIES for severity in SEVERITIES]
    raise ValueError(name)


def averaged(archive, keys):
    missing = set(keys).difference(archive.files)
    if missing:
        raise SystemExit(f"probe archive is missing keys: {sorted(missing)}")
    return np.nanmean(np.stack([archive[key] for key in keys]), axis=0)


def paired_blocks(
    runs, seeds, descriptor, weight, name,
    reference_descriptor="global", reference_weight=0,
):
    blocks = []
    for seed in seeds:
        with np.load(runs[seed] / "probe_outcomes.npz") as archive:
            labels = archive["probe_labels"]
            before = averaged(
                archive, endpoint_keys(reference_descriptor, reference_weight, name)
            )
            after = averaged(archive, endpoint_keys(descriptor, weight, name))
            blocks.append([
                float(np.nanmean(after[labels == identity] - before[labels == identity]))
                for identity in np.unique(labels)
            ])
    return np.asarray(blocks)


def bootstrap(blocks, samples, rng):
    seed_count, identity_count = blocks.shape
    seed_draws = rng.integers(0, seed_count, size=(samples, seed_count))
    identity_draws = rng.integers(0, identity_count, size=(samples, seed_count, identity_count))
    draws = np.take_along_axis(blocks[seed_draws], identity_draws, axis=2).mean(axis=(1, 2))
    return {
        "difference": float(blocks.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [.025, .975])],
        "probability_positive": float(np.mean(draws > 0)),
    }


def write_summary(path, rows):
    fields = ["descriptor", "fusion_weight", "protocol", "corruption", "severity",
              "rank1_mean", "rank1_std", "rank1_count"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    runs, design = discover(args.results, seeds)
    by_seed = result_rows(runs, seeds)
    summary = aggregate_rows(by_seed, seeds)
    output = args.output or args.results
    output.mkdir(parents=True, exist_ok=True)
    write_summary(output / "summary.csv", summary)

    global_result = aggregate_endpoints(by_seed, seeds, "global", 0)
    descriptors = sorted({row["descriptor"] for row in summary if row["descriptor"] != "global"})
    sensitivity = {}
    for descriptor in descriptors:
        sensitivity[descriptor] = {}
        for weight in design["fusion_weights"]:
            current = aggregate_endpoints(by_seed, seeds, descriptor, float(weight))
            sensitivity[descriptor][str(weight)] = {
                **current,
                "clean_difference": current["clean"] - global_result["clean"],
                "robust_difference": current["robust"] - global_result["robust"],
                "families_improved": sum(
                    current[f"family:{family}"] > global_result[f"family:{family}"]
                    for family in FAMILIES
                ),
            }

    rng = np.random.default_rng(args.bootstrap_seed)
    paired = {}
    for descriptor in DETERMINISTIC + ["frozen_head_reliable"]:
        paired[descriptor] = {
            name: bootstrap(
                paired_blocks(runs, seeds, descriptor, PRIMARY_WEIGHT, name),
                args.bootstrap_samples, rng,
            )
            for name in ("clean", "robust", "severity_05", "occluded")
        }
    paired["raw_reliable_minus_uniform"] = {
        name: bootstrap(
            paired_blocks(
                runs, seeds, "raw_reliable", PRIMARY_WEIGHT, name,
                reference_descriptor="raw_uniform", reference_weight=PRIMARY_WEIGHT,
            ),
            args.bootstrap_samples, rng,
        )
        for name in ("clean", "robust", "severity_05", "occluded")
    }

    decisions = {}
    for descriptor in DETERMINISTIC:
        current = sensitivity[descriptor][str(PRIMARY_WEIGHT)]
        decisions[descriptor] = {
            "primary_fusion_weight": PRIMARY_WEIGHT,
            "clean_difference": current["clean_difference"],
            "robust_difference": current["robust_difference"],
            "families_improved": current["families_improved"],
            "passes": bool(
                current["clean_difference"] >= -.005
                and current["robust_difference"] >= .02
                and current["families_improved"] >= 3
            ),
        }
    random_primary = [
        sensitivity[name][str(PRIMARY_WEIGHT)]
        for name in descriptors if name.startswith("random_head_")
    ]
    random_summary = {
        "projection_seeds": len(random_primary),
        "clean_difference_range": [
            min(row["clean_difference"] for row in random_primary),
            max(row["clean_difference"] for row in random_primary),
        ],
        "robust_difference_range": [
            min(row["robust_difference"] for row in random_primary),
            max(row["robust_difference"] for row in random_primary),
        ],
        "passing_projection_seeds": sum(
            row["clean_difference"] >= -.005 and row["robust_difference"] >= .02
            and row["families_improved"] >= 3 for row in random_primary
        ),
    }
    payload = {
        "design": {
            "training_seeds": seeds, "random_head_seeds": design["random_head_seeds"],
            "fusion_weights": design["fusion_weights"], "primary_fusion_weight": PRIMARY_WEIGHT,
        },
        "global": global_result,
        "sensitivity": sensitivity,
        "paired_bootstrap": paired,
        "acceptance_gate": decisions,
        "random_head_stability": random_summary,
        "recommendation": "advance_to_p1" if any(row["passes"] for row in decisions.values()) else "stop_part_method",
    }
    (output / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"runs": len(runs), "recommendation": payload["recommendation"], "gate": decisions}, indent=2))


if __name__ == "__main__":
    main()
