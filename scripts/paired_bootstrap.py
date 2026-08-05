#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Identity-level paired bootstrap for two completed runs")
    parser.add_argument("baseline", type=Path, help="baseline run directory")
    parser.add_argument("candidate", type=Path, help="candidate run directory")
    parser.add_argument("--key", default="official_condition_OCC")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    with np.load(args.baseline / "probe_outcomes.npz") as base, np.load(args.candidate / "probe_outcomes.npz") as candidate:
        if not np.array_equal(base["probe_indices"], candidate["probe_indices"]):
            raise SystemExit("runs do not contain the same probe ordering")
        if not np.array_equal(base["probe_labels"], candidate["probe_labels"]):
            raise SystemExit("runs do not contain the same probe labels")
        if args.key not in base.files or args.key not in candidate.files:
            raise SystemExit(f"outcome key {args.key!r} not present in both runs")
        labels = base["probe_labels"]
        base_outcome, candidate_outcome = base[args.key], candidate[args.key]
    selected = np.isfinite(base_outcome) & np.isfinite(candidate_outcome)
    labels, base_outcome, candidate_outcome = labels[selected], base_outcome[selected], candidate_outcome[selected]
    if not len(labels):
        raise SystemExit("the selected outcome has no paired finite probes")
    identities = np.unique(labels)
    identity_differences = np.array([
        candidate_outcome[labels == identity].mean() - base_outcome[labels == identity].mean()
        for identity in identities
    ])
    rng = np.random.default_rng(args.seed)
    draws = rng.choice(identity_differences, (args.samples, len(identity_differences)), replace=True).mean(axis=1)
    result = {
        "baseline_run": args.baseline.name,
        "candidate_run": args.candidate.name,
        "outcome": args.key,
        "identities": len(identities),
        "observed_difference": float(identity_differences.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_samples": args.samples,
        "seed": args.seed,
    }
    output = args.candidate / f"paired_bootstrap_{args.key}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
