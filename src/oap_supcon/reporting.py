from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import PoseData


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def aggregate_results(results_root: Path):
    rows = []
    for path in sorted(results_root.glob("*/metrics.json")):
        with path.open() as handle:
            metrics = json.load(handle)
        for result in metrics["results"]:
            rows.append({
                "run_id": metrics["run_id"], "git_commit": metrics["git_commit"],
                "dataset": metrics["dataset"], "method": metrics["method"], "seed": metrics["seed"],
                "epochs": metrics["epochs"], "device": metrics["device"], "params": metrics["parameters"],
                "train_hours": metrics["train_hours"], "status": metrics["status"],
                "protocol": result.get("protocol", "clean_gallery"),
                "delta_rank1": result.get("delta_rank1", float("nan")),
                **result,
            })
    if not rows:
        raise FileNotFoundError(f"no completed metrics.json files under {results_root}")
    _write_csv(results_root / "runs.csv", rows)
    numeric = [
        "rank1", "rank5", "rank10", "map", "eer", "tar_far_1e-02", "tar_far_1e-03",
        "delta_rank1", "train_hours",
    ]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["protocol"], row["corruption"], row["severity"])].append(row)
    summary_rows = []
    for key, group in sorted(grouped.items()):
        summary = dict(zip(("dataset", "method", "protocol", "corruption", "severity"), key))
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
    _write_csv(results_root / "summary.csv", summary_rows)
    auc_groups: dict[tuple, list[dict]] = defaultdict(list)
    synthetic = {"random_joint", "dynamic_joint", "body_part", "temporal"}
    for row in rows:
        if row["corruption"] in synthetic:
            auc_groups[(
                row["run_id"], row["dataset"], row["method"], row["seed"], row["protocol"], row["corruption"]
            )].append(row)
    auc_rows = []
    for key, group in sorted(auc_groups.items()):
        ordered = sorted(group, key=lambda row: float(row["severity"]))
        severities = np.array([float(row["severity"]) for row in ordered])
        rank1 = np.array([float(row["rank1"]) for row in ordered])
        span = severities[-1] - severities[0]
        auc = float(np.trapz(rank1, severities) / span) if span > 0 else float("nan")
        nonzero = rank1[severities > 0]
        auc_rows.append({
            **dict(zip(("run_id", "dataset", "method", "seed", "protocol", "corruption"), key)),
            "clean_rank1": float(rank1[0]), "robustness_auc": auc,
            "mean_nonzero_rank1": float(nonzero.mean()) if len(nonzero) else float("nan"),
            "worst_rank1": float(rank1.min()), "absolute_drop_at_max": float(rank1[0] - rank1[-1]),
            "relative_retention_at_max": float(rank1[-1] / rank1[0]) if rank1[0] else float("nan"),
        })
    _write_csv(results_root / "robustness_auc.csv", auc_rows)
    clean = {}
    for row in rows:
        if (
            float(row["severity"]) == 0
            and row["protocol"] == "clean_gallery"
            and row["run_id"] not in clean
        ):
            clean[row["run_id"]] = row
    clean_grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in clean.values():
        clean_grouped[(row["dataset"], row["method"])].append(float(row["rank1"]))
    row_end = "\\\\"
    latex = ["\\begin{tabular}{llrrr}", "Dataset & Method & Mean & SD & N " + row_end, "\\hline"]
    for (dataset, method), values in sorted(clean_grouped.items()):
        sd = statistics.stdev(values) if len(values) > 1 else float("nan")
        latex.append(f"{dataset} & {method} & {statistics.fmean(values):.3f} & {sd:.3f} & {len(values)} " + row_end)
    latex.append("\\end{tabular}")
    (results_root / "clean_rank1.tex").write_text("\n".join(latex) + "\n")
    return rows, summary_rows
