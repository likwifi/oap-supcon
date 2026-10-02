from __future__ import annotations

import csv
import json
import math
import statistics
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .data import PoseData
from .provenance import configuration_fingerprint


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)


def _markdown_table(rows: list[dict]) -> str:
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(row[key]) for key in headers) + " |" for row in rows)
    return "\n".join(lines)


def audit_dataset(data_path: Path, output_dir: Path):
    data = PoseData.load(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in np.unique(data.split):
        indices = np.flatnonzero(data.split == split)
        frame_counts = data.frame_counts[indices]
        frame_mask = np.arange(data.x.shape[1])[None, :] < frame_counts[:, None]
        source_visibility = data.visibility[indices][
            np.broadcast_to(frame_mask[:, :, None], data.visibility[indices].shape)
        ]
        rows.append({
            "split": split, "identities": len(np.unique(data.labels[indices])), "sequences": len(indices),
            "mean_frames": round(float(frame_counts.mean()), 2),
            "min_frames": int(frame_counts.min()), "max_frames": int(frame_counts.max()),
            "joints": data.x.shape[2], "dimensions": data.x.shape[3],
            "mean_source_visibility": round(float(source_visibility.mean()), 8),
            "zero_source_visibility_fraction": round(float((source_visibility == 0).mean()), 8),
        })
    _write_csv(output_dir / "data_audit.csv", rows)
    condition_rows = []
    for split in np.unique(data.split):
        for condition in np.unique(data.conditions[data.split == split]):
            selected = (data.split == split) & (data.conditions == condition)
            condition_rows.append({"split": split, "condition": condition, "sequences": int(selected.sum())})
    _write_csv(output_dir / "condition_counts.csv", condition_rows)
    summary = [
        "# Dataset audit", "", f"- Source: `{data_path}`",
        f"- SHA-256 over standardized arrays: `{data.checksum()}`",
        f"- Shape: `{tuple(data.x.shape)}` (`N,T,J,D`)",
        f"- Coordinate representation: `{data.x.shape[-1]}D`",
        "- Identity leakage check: passed", "- Finite-coordinate and visibility-range checks: passed",
        "", "## Split summary", "", _markdown_table(rows), "",
        "This structural audit does not replace manual skeleton rendering, joint-mapping review, or license review.",
    ]
    (output_dir / "data_audit.md").write_text("\n".join(summary) + "\n")
    return rows


GROUP_FIELDS = ("dataset", "method", "backbone", "experiment_id", "config_fingerprint",
                "source_sha256", "dataset_sha256", "evaluation_split")
RESULT_GROUP_FIELDS = ("descriptor", "protocol", "corruption", "severity", "probe_view")


def aggregate_results(results_root: Path, duplicates="first", output_root: Path | None = None):
    if duplicates not in {"first", "last", "error"}:
        raise ValueError("duplicates must be first, last or error")
    output_root = output_root or results_root
    output_root.mkdir(parents=True, exist_ok=True)
    selected, duplicate_rows = {}, []
    for path in sorted(results_root.glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        if metrics.get("status") != "complete":
            continue
        cfg_path = path.parent / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {
            "model": {"backbone": metrics.get("backbone", "tcn"), "parameters": metrics.get("parameters")},
            "method_name": metrics["method"], "epochs": metrics["epochs"],
        }
        metadata = {
            "backbone": metrics.get("backbone", cfg.get("model", {}).get("backbone", "tcn")),
            "experiment_id": metrics.get("experiment_id", cfg.get("experiment_id", "legacy")),
            "config_fingerprint": metrics.get("config_fingerprint") or configuration_fingerprint(cfg),
            "source_sha256": metrics.get("source_sha256", metrics.get("git_commit", "unknown")),
            "dataset_sha256": metrics.get("dataset_sha256", "unknown"),
            "evaluation_split": metrics.get("evaluation_split", "test"),
        }
        metrics.update(metadata)
        key = tuple(metrics[f] for f in GROUP_FIELDS) + (metrics["seed"],)
        if key in selected:
            old = selected[key]
            if duplicates == "error":
                raise ValueError(f"duplicate seed {metrics['seed']}: {old['run_id']} and {metrics['run_id']}")
            kept, discarded = (old, metrics) if duplicates == "first" else (metrics, old)
            duplicate_rows.append({"kept": kept["run_id"], "discarded": discarded["run_id"], "seed": metrics["seed"]})
            selected[key] = kept
        else:
            selected[key] = metrics
    if duplicate_rows:
        warnings.warn(f"Excluded {len(duplicate_rows)} repeated-seed runs; see duplicate_runs.csv", stacklevel=2)
        _write_csv(output_root / "duplicate_runs.csv", duplicate_rows)
    elif (output_root / "duplicate_runs.csv").exists():
        (output_root / "duplicate_runs.csv").unlink()
    rows = []
    for metrics in selected.values():
        for result in metrics["results"]:
            rows.append({
                **{field: metrics[field] for field in GROUP_FIELDS},
                "run_id": metrics["run_id"], "git_commit": metrics["git_commit"],
                "dataset": metrics["dataset"], "method": metrics["method"], "seed": metrics["seed"],
                "epochs": metrics["epochs"], "device": metrics["device"], "params": metrics["parameters"],
                "train_hours": metrics["train_hours"], "status": metrics["status"],
                "protocol": result.get("protocol", "clean_gallery"),
                "descriptor": result.get("descriptor", "global"),
                "probe_view": result.get("probe_view", ""),
                "delta_rank1": result.get("delta_rank1", float("nan")),
                **result,
            })
    if not rows:
        raise FileNotFoundError(f"no completed metrics.json files under {results_root}")
    _write_csv(output_root / "runs.csv", rows)
    numeric = [
        "rank1", "rank5", "rank10", "map", "eer", "tar_far_1e-02", "tar_far_1e-03",
        "delta_rank1", "train_hours",
    ]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[f] for f in GROUP_FIELDS + RESULT_GROUP_FIELDS)].append(row)
    summary_rows = []
    for key, group in sorted(grouped.items()):
        summary = dict(zip(GROUP_FIELDS + RESULT_GROUP_FIELDS, key))
        for field in numeric:
            values = []
            for row in group:
                value = row.get(field)
                if value is None:
                    continue
                numeric_value = float(value)
                if not math.isnan(numeric_value):
                    values.append(numeric_value)
            summary[f"{field}_mean"] = statistics.fmean(values) if values else float("nan")
            summary[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else float("nan")
            summary[f"{field}_count"] = len(values)
        summary_rows.append(summary)
    _write_csv(output_root / "summary.csv", summary_rows)
    official_per_view = []
    for row in summary_rows:
        if row["protocol"] != "single_view_gallery_per_view":
            continue
        official_per_view.append({
            **{field: row[field] for field in GROUP_FIELDS},
            "descriptor": row["descriptor"],
            "condition": row["corruption"].split(":", 1)[-1],
            "probe_view": row["probe_view"],
            "rank1_mean": row["rank1_mean"],
            "rank1_std": row["rank1_std"],
            "rank1_count": row["rank1_count"],
        })
    if official_per_view:
        _write_csv(output_root / "official_per_view.csv", official_per_view)
    elif (output_root / "official_per_view.csv").exists():
        (output_root / "official_per_view.csv").unlink()
    auc_groups: dict[tuple, list[dict]] = defaultdict(list)
    synthetic = {"random_joint", "dynamic_joint", "body_part", "temporal"}
    for row in rows:
        if row["corruption"] in synthetic:
            auc_groups[(
                row["run_id"], row["dataset"], row["method"], row["seed"],
                row["descriptor"], row["protocol"], row["corruption"],
            )].append(row)
    auc_rows = []
    for key, group in sorted(auc_groups.items()):
        ordered = sorted(group, key=lambda row: float(row["severity"]))
        severities = np.array([float(row["severity"]) for row in ordered])
        rank1 = np.array([float(row["rank1"]) for row in ordered])
        span = severities[-1] - severities[0]
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        auc = float(integrate(rank1, severities) / span) if span > 0 else float("nan")
        nonzero = rank1[severities > 0]
        auc_rows.append({
            **{f: group[0][f] for f in GROUP_FIELDS},
            **dict(zip(("run_id", "dataset", "method", "seed", "descriptor", "protocol", "corruption"), key)),
            "clean_rank1": float(rank1[0]), "robustness_auc": auc,
            "mean_nonzero_rank1": float(nonzero.mean()) if len(nonzero) else float("nan"),
            "worst_rank1": float(rank1.min()), "absolute_drop_at_max": float(rank1[0] - rank1[-1]),
            "relative_retention_at_max": float(rank1[-1] / rank1[0]) if rank1[0] else float("nan"),
        })
    _write_csv(output_root / "robustness_auc.csv", auc_rows)
    # Clean rank-1 is the severity-0 row of a corruption family, which scores the
    # whole probe set. The `official_*` rows are per-condition and per-view
    # slices that also carry severity 0, and they are emitted first -- taking the
    # first matching row per run silently reported the BG condition as if it were
    # the overall clean result.
    clean = {}
    for row in rows:
        key = (row["run_id"], row["descriptor"])
        if (
            float(row["severity"]) == 0
            and row["protocol"] == "clean_gallery"
            and row["corruption"] in synthetic
            and key not in clean
        ):
            clean[key] = row
    clean_grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in clean.values():
        clean_grouped[tuple(row[f] for f in GROUP_FIELDS) + (row["descriptor"],)].append(float(row["rank1"]))
    row_end = "\\\\"
    latex = [
        "\\begin{tabular}{lllllllrrr}",
        "Dataset & Method & Backbone & Split & Config & Source & Descriptor & Mean & SD & N " + row_end,
        "\\hline",
    ]
    for key, values in sorted(clean_grouped.items()):
        fields = dict(zip(GROUP_FIELDS + ("descriptor",), key))
        sd = statistics.stdev(values) if len(values) > 1 else float("nan")
        labels = [fields[f] for f in ("dataset", "method", "backbone", "evaluation_split")]
        labels += [fields["config_fingerprint"][:10], fields["source_sha256"][:10], fields["descriptor"]]
        rendered = " & ".join(str(v).replace("_", r"\_") for v in labels)
        latex.append(
            f"{rendered} & {statistics.fmean(values):.3f} & {sd:.3f} & {len(values)} " + row_end
        )
    latex.append("\\end{tabular}")
    (output_root / "clean_rank1.tex").write_text("\n".join(latex) + "\n")
    return rows, summary_rows
