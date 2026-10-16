#!/usr/bin/env python3
"""Stratify matched Gait3D descriptor outcomes by real pose-estimator evidence.

Gait3D has no occlusion annotation. Each scored probe is therefore placed in
quantile bins of a proxy computed only from its released HRNet confidences,
over the same real frames the evaluator embeds. No model is re-run: per-probe
clean-gallery outcomes come from `probe_outcomes.npz`.

Predeclared primary endpoint: the height-matched interaction. Probes are split
into skeleton-height tertiles; within each, into tertiles of
`low_confidence_fraction` (joint-frames with confidence below 0.5). In every
height stratum, the `raw_reliable` minus `global` Rank-1 gain among its most
low-confidence probes minus that gain among its least is computed, and the three
strata are averaged with equal weight. Matching on height keeps small, distant
people from standing in for occluded ones. The same interaction over global
tertiles, without height matching, is the secondary endpoint. The 0.5 threshold
and both endpoints were fixed from the confidence and height distributions
before any outcome was inspected: at 0.3, a fifth of probes have no qualifying
joint-frame and tertiles tie. Every other proxy, contrast and stratum is
exploratory.

Intervals resample training seeds and probe identities as crossed clusters; the
same draws are shared by every bin, descriptor and contrast, so differences are
paired. Low confidence also follows resolution and blur, so per-bin sequence
length and skeleton height are reported as confounders. A blinded manual-audit
sheet supports checking whether the proxy tracks visible occlusion.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_gait3d import DESCRIPTORS, discover_runs  # noqa: E402
from oap_supcon.augment import anatomical_parts  # noqa: E402
from oap_supcon.data import PoseData, development_split, resolve_dataset_path  # noqa: E402

PRIMARY_PROXY = "low_confidence_fraction"
PRIMARY_CONTRAST = "raw_reliable_minus_global"
CONTRASTS = (("raw_reliable", "global"), ("raw_uniform", "global"), ("raw_reliable", "raw_uniform"))
PROXIES = {
    "low_confidence_fraction": "fraction of real joint-frames with HRNet confidence < 0.5",
    "low_confidence_fraction_0.3": "fraction of real joint-frames with HRNet confidence < 0.3",
    "confidence_deficit": "1 - mean HRNet confidence over real joint-frames",
    "worst_part_deficit": "1 - lowest mean confidence among the five anatomical parts",
}


def probe_proxies(data: PoseData, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Occlusion proxies and confounders over each probe's real frames."""
    parts = anatomical_parts(data.x.shape[2])
    columns = {name: [] for name in (*PROXIES, "frames", "skeleton_height_px")}
    for index in indices:
        length = int(data.frame_counts[index])
        confidence = data.visibility[index, :length]
        heights = np.ptp(data.x[index, :length, :, 1], axis=1)
        columns["low_confidence_fraction"].append(float(np.mean(confidence < .5)))
        columns["low_confidence_fraction_0.3"].append(float(np.mean(confidence < .3)))
        columns["confidence_deficit"].append(float(1 - confidence.mean()))
        columns["worst_part_deficit"].append(float(1 - min(confidence[:, part].mean() for part in parts)))
        columns["frames"].append(length)
        # Crop-pixel units; a resolution covariate, not a calibrated body height.
        columns["skeleton_height_px"].append(float(np.median(heights)))
    return {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


def quantile_bins(values: np.ndarray, bins: int):
    """Bin 0 holds the least occlusion evidence. Ties stay together."""
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    assignment = np.searchsorted(edges[1:-1], values, side="right")
    counts = np.bincount(assignment, minlength=bins)
    if np.any(counts == 0):
        raise ValueError(f"empty quantile bin (counts {counts.tolist()}); proxy has too many ties")
    return assignment, edges


def bin_labels(bins: int) -> list[str]:
    return ["low", "middle", "high"] if bins == 3 else [f"q{number + 1}" for number in range(bins)]


def spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    def ranks(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        starts = np.cumsum(counts) - counts
        return (starts + (counts - 1) / 2)[inverse]
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return None  # undefined for a constant variable; JSON has no NaN
    return float(np.corrcoef(ranks(first), ranks(second))[0, 1])


def clean_outcomes(archive, descriptor: str) -> np.ndarray:
    """All severity-0 clean-gallery keys score the same uncorrupted probes."""
    keys = sorted(key for key in archive.files
                  if key.startswith(f"{descriptor}_clean_gallery_") and key.endswith("_0"))
    if not keys:
        raise ValueError(f"no clean-gallery outcomes for {descriptor}")
    outcome = archive[keys[0]].astype(np.float64)
    if any(not np.array_equal(archive[key], archive[keys[0]]) for key in keys[1:]):
        raise ValueError(f"clean-gallery outcomes disagree across keys for {descriptor}")
    if not np.isin(outcome, (0, 1)).all():
        raise ValueError(f"clean outcomes for {descriptor} must be binary per probe")
    return outcome


def cluster_sums(outcomes: np.ndarray, identities: np.ndarray, assignment: np.ndarray, bins: int):
    """Per-seed, per-identity correct counts in each bin plus a pooled final column."""
    clusters = np.unique(identities, return_inverse=True)[1]
    sums = np.zeros((outcomes.shape[0], clusters.max() + 1, bins + 1))
    counts = np.zeros((clusters.max() + 1, bins + 1))
    for column in (assignment, np.full_like(assignment, bins)):
        np.add.at(counts, (clusters, column), 1)
        for seed in range(outcomes.shape[0]):
            np.add.at(sums[seed], (clusters, column), outcomes[seed])
    return sums, counts


def bootstrap_rank1(sums: dict, counts: np.ndarray, samples: int, rng, chunk: int = 1000):
    """Crossed seed x identity draws, shared across descriptors and bins."""
    seeds, clusters = next(iter(sums.values())).shape[:2]
    draws = {name: [] for name in sums}
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        seed_weight = rng.multinomial(seeds, np.full(seeds, 1 / seeds), size=size)
        identity_weight = rng.multinomial(clusters, np.full(clusters, 1 / clusters), size=size)
        denominator = seeds * identity_weight @ counts
        for name, value in sums.items():
            per_seed = np.einsum("bc,sck->bsk", identity_weight, value)
            numerator = np.einsum("bs,bsk->bk", seed_weight, per_seed)
            with np.errstate(invalid="ignore", divide="ignore"):
                draws[name].append(numerator / denominator)
    return {name: np.concatenate(values) for name, values in draws.items()}


def summarize(point: float, draws: np.ndarray) -> dict:
    finite = draws[np.isfinite(draws)]
    return {"difference": float(point), "ci95_low": float(np.quantile(finite, .025)),
            "ci95_high": float(np.quantile(finite, .975)),
            "positive_fraction": float(np.mean(finite > 0)), "valid_draws": int(finite.size)}


def height_matched(outcomes, identities, proxies, bins, samples, rng):
    """Occlusion contrasts within skeleton-height strata, averaged with equal weight."""
    height = quantile_bins(proxies["skeleton_height_px"], bins)[0]
    within = np.empty(len(height), dtype=int)
    for stratum in range(bins):
        members = height == stratum
        within[members] = quantile_bins(proxies[PRIMARY_PROXY][members], bins)[0]
    cells = bins * bins
    sums, counts = {}, None
    for descriptor in DESCRIPTORS:
        sums[descriptor], counts = cluster_sums(outcomes[descriptor], identities, height * bins + within, cells)
    point = {d: value.sum(axis=(0, 1)) / (outcomes[d].shape[0] * counts.sum(0)) for d, value in sums.items()}
    draws = bootstrap_rank1(sums, counts, samples, rng)

    def matched(values):
        grid = values[..., :cells].reshape(*values.shape[:-1], bins, bins)
        return (grid[..., bins - 1] - grid[..., 0]).mean(axis=-1)

    rows = [{"proxy": PRIMARY_PROXY, "contrast": f"{d}_high_minus_low", "scope": "height_matched_occlusion_effect",
             "role": "exploratory", **summarize(matched(point[d]), matched(draws[d]))} for d in DESCRIPTORS]
    for candidate, baseline in CONTRASTS:
        name = f"{candidate}_minus_{baseline}"
        rows.append({"proxy": PRIMARY_PROXY, "contrast": name,
                     "scope": "height_matched_interaction_high_minus_low",
                     "role": "primary" if name == PRIMARY_CONTRAST else "exploratory",
                     **summarize(matched(point[candidate] - point[baseline]),
                                 matched(draws[candidate] - draws[baseline]))})
    return rows


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_sheets(output: Path, proxies, assignment, labels, data, indices, per_bin, rng):
    """Blinded sheet for annotators, and a separate key holding the proxy bins."""
    chosen = []
    for number in range(len(labels)):
        members = np.flatnonzero(assignment == number)
        chosen.extend(rng.choice(members, min(per_bin, len(members)), replace=False).tolist())
    chosen = rng.permutation(chosen)
    sheet, key = [], []
    for item, position in enumerate(chosen, 1):
        index = int(indices[position])
        common = {"item": item, "sequence_id": str(data.sequence_ids[index]),
                  "camera": str(data.views[index]), "frames": int(data.frame_counts[index])}
        sheet.append({**common, "occluded": "", "occluded_fraction_of_frames": "",
                      "occluder": "", "notes": ""})
        key.append({**common, "probe_index": index, "primary_bin": labels[assignment[position]],
                    PRIMARY_PROXY: proxies[PRIMARY_PROXY][position]})
    write_csv(output / "manual_audit_sheet.csv", sheet)
    write_csv(output / "manual_audit_key.csv", key)


def analyze(root: Path, output: Path, seeds: list[int], samples: int, bootstrap_seed: int, *,
            dataset: Path | None = None, run_names: list[str] | None = None, bins: int = 3,
            audit_per_bin: int = 30):
    if samples < 1 or bins < 2 or audit_per_bin < 0:
        raise ValueError("bootstrap samples, bins >= 2 and audit size must be valid")
    runs = discover_runs(root, seeds, run_names)
    first_directory, first = runs[seeds[0]]
    cfg = yaml.safe_load((first_directory / "config.yaml").read_text())
    if dataset is None:
        recorded = Path(cfg.get("dataset_path", ""))
        dataset = recorded if recorded.is_file() else resolve_dataset_path(ROOT, cfg["dataset"])
    data, _ = development_split(PoseData.load(dataset), cfg)
    if data.checksum() != first["dataset_sha256"]:
        raise ValueError("dataset or its validation split does not match the runs; pass --dataset")
    stage = first["evaluation_split"]
    expected = data.indices(["val_probe"] if stage == "validation" else ["probe", "test"])

    outcomes = {descriptor: [] for descriptor in DESCRIPTORS}
    for seed in seeds:
        with np.load(runs[seed][0] / "probe_outcomes.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["probe_indices"], expected):
                raise ValueError(f"seed {seed} probes differ from the {stage} probes of this dataset")
            if not np.array_equal(archive["probe_labels"], data.labels[expected]):
                raise ValueError(f"seed {seed} probe labels differ from the dataset")
            for descriptor in DESCRIPTORS:
                outcomes[descriptor].append(clean_outcomes(archive, descriptor))
    outcomes = {descriptor: np.stack(values) for descriptor, values in outcomes.items()}

    proxies = probe_proxies(data, expected)
    identities = data.labels[expected]
    labels = bin_labels(bins)
    scopes = [*labels, "all"]
    rng = np.random.default_rng(bootstrap_seed)
    output.mkdir(parents=True, exist_ok=True)
    strata, contrasts, edges_by_proxy, assignments, skipped = [], [], {}, {}, {}
    for proxy in PROXIES:
        try:
            assignment, edges = quantile_bins(proxies[proxy], bins)
        except ValueError as error:
            if proxy == PRIMARY_PROXY:
                raise
            skipped[proxy] = str(error)
            continue
        assignments[proxy], edges_by_proxy[proxy] = assignment, edges.tolist()
        sums, counts = {}, None
        for descriptor in DESCRIPTORS:
            sums[descriptor], counts = cluster_sums(outcomes[descriptor], identities, assignment, bins)
        point = {d: value.sum(axis=(0, 1)) / (len(seeds) * counts.sum(0)) for d, value in sums.items()}
        draws = bootstrap_rank1(sums, counts, samples, rng)
        for k, scope in enumerate(scopes):
            members = assignment == k if k < bins else np.ones(len(expected), dtype=bool)
            for descriptor in DESCRIPTORS:
                interval = summarize(point[descriptor][k], draws[descriptor][:, k])
                strata.append({
                    "proxy": proxy, "stratum": scope, "descriptor": descriptor,
                    "probes": int(members.sum()), "identities": int(np.unique(identities[members]).size),
                    "proxy_min": float(proxies[proxy][members].min()),
                    "proxy_max": float(proxies[proxy][members].max()),
                    "mean_frames": float(proxies["frames"][members].mean()),
                    "median_skeleton_height_px": float(np.median(proxies["skeleton_height_px"][members])),
                    "rank1": interval["difference"], "rank1_ci95_low": interval["ci95_low"],
                    "rank1_ci95_high": interval["ci95_high"],
                    **{f"rank1_seed{seed}": float(outcomes[descriptor][s, members].mean())
                       for s, seed in enumerate(seeds)},
                })
        high, low = bins - 1, 0
        for descriptor in DESCRIPTORS:
            contrasts.append({"proxy": proxy, "contrast": f"{descriptor}_high_minus_low", "scope": "occlusion_effect",
                              "role": "exploratory", **summarize(
                                  point[descriptor][high] - point[descriptor][low],
                                  draws[descriptor][:, high] - draws[descriptor][:, low])})
        for candidate, baseline in CONTRASTS:
            name = f"{candidate}_minus_{baseline}"
            gain_point = point[candidate] - point[baseline]
            gain_draws = draws[candidate] - draws[baseline]
            for k, scope in enumerate(scopes):
                contrasts.append({"proxy": proxy, "contrast": name, "scope": scope, "role": "exploratory",
                                  **summarize(gain_point[k], gain_draws[:, k])})
            secondary = proxy == PRIMARY_PROXY and name == PRIMARY_CONTRAST
            contrasts.append({"proxy": proxy, "contrast": name, "scope": "interaction_high_minus_low",
                              "role": "secondary" if secondary else "exploratory",
                              **summarize(gain_point[high] - gain_point[low],
                                          gain_draws[:, high] - gain_draws[:, low])})

    contrasts.extend(height_matched(outcomes, identities, proxies, bins, samples, rng))
    write_csv(output / "occlusion_strata.csv", strata)
    write_csv(output / "occlusion_contrasts.csv", contrasts)
    write_csv(output / "probe_occlusion_proxies.csv", [
        {"probe_index": int(index), "sequence_id": str(data.sequence_ids[index]),
         "identity": int(data.labels[index]), "camera": str(data.views[index]),
         **{name: float(values[position]) for name, values in proxies.items()},
         **{f"{proxy}_bin": labels[assignments[proxy][position]] for proxy in assignments},
         **{f"{d}_correct_seeds": int(outcomes[d][:, position].sum()) for d in DESCRIPTORS}}
        for position, index in enumerate(expected)])
    if audit_per_bin:
        audit_sheets(output, proxies, assignments[PRIMARY_PROXY], labels, data, expected,
                     audit_per_bin, np.random.default_rng(bootstrap_seed + 1))
    primary = next(row for row in contrasts if row["role"] == "primary")
    result = {
        "dataset": "gait3d", "evaluation_split": stage, "seeds": seeds,
        "runs": {str(seed): runs[seed][0].name for seed in seeds},
        "dataset_path": str(dataset), "dataset_sha256": first["dataset_sha256"],
        "source_coverage": first.get("source_coverage"), "probes": int(len(expected)),
        "outcome": "clean-gallery Rank-1 of uncorrupted probes; no synthetic masks",
        "proxies": PROXIES, "primary": {"proxy": PRIMARY_PROXY, "contrast": PRIMARY_CONTRAST, **primary},
        "secondary": next(row for row in contrasts if row["role"] == "secondary"),
        "bins": bins, "bin_labels": labels, "bin_edges": edges_by_proxy,
        "skipped_proxies": skipped,
        "confounder_spearman": {proxy: {"frames": spearman(proxies[proxy], proxies["frames"]),
                                        "skeleton_height_px": spearman(proxies[proxy], proxies["skeleton_height_px"])}
                                for proxy in assignments},
        "bootstrap_samples": samples, "bootstrap_seed": bootstrap_seed,
        "limitations": [
            "Proxies measure low pose-estimator confidence, not annotated occlusion.",
            "Low confidence also follows resolution and blur; compare per-bin skeleton height.",
            "Bins use probe evidence only; gallery tracks may also be occluded.",
            "Intervals are pointwise and unadjusted except for the single primary endpoint.",
        ],
    }
    (output / "occlusion_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Gait3D results root")
    parser.add_argument("--runs", nargs="+", help="run directory names under RESULTS, one per seed")
    parser.add_argument("--dataset", type=Path, help="Gait3D dataset.npz; defaults to the run's recorded path")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--audit-per-bin", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261008)
    args = parser.parse_args()
    result = analyze(args.results, args.output or args.results / "occlusion_strata",
                     [int(v) for v in args.seeds.split(",")], args.bootstrap_samples, args.bootstrap_seed,
                     dataset=args.dataset, run_names=args.runs, bins=args.bins,
                     audit_per_bin=args.audit_per_bin)
    print(json.dumps({key: result[key] for key in ("evaluation_split", "probes", "primary", "confounder_spearman")},
                     indent=2))


if __name__ == "__main__":
    main()
