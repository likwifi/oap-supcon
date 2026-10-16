#!/usr/bin/env python3
"""Collect every number used by the Pattern Recognition manuscript; no training.

Run from the project root with an interpreter that has NumPy, PyYAML and
Matplotlib (on the cluster: .venv_plot/bin/python). Result artifacts are read
only. The script checks that each cohort has the expected seeds, probes and
dataset/source hashes, then writes evidence_pr.json, fig1.pdf (test-set summary), fig3.pdf
(descriptor effect across cohorts), fig4.pdf (CASIA-B five-seed severity curves) and the
LaTeX table bodies in tables/.

Inputs (all produced by the project's own analysis scripts):
  results_casia_validation_v2/comparison           compare_runs.py, 7 objectives x 5 seeds
  results_descriptor_controls_5seed/analysis.json  analyze_descriptor_controls.py
  results_gait3d_v2_comparison/comparison          compare_runs.py, 4 objectives x 3 seeds
  results_descriptor_controls_gait3d/analysis.json analyze_descriptor_controls.py
  results_gait3d_v2_comparison_final/comparison    compare_runs.py, 2 objectives x 5 seeds (test)
  results_gait3d_v2_comparison/occlusion_strata    analyze_gait3d_occlusion.py
  results_*/actual_missingness                     analyze_actual_missingness.py
  logs/fastposegait_*.out, logs/oap_efficiency_*.out
  results_fpg_part_matching/                       fpg_part_matching.py, GPGait transfer test
  Paper/IJCV/evidence_crossed.json                 three-seed factorial (unchanged)
"""
from __future__ import annotations

import csv
import hashlib
import os
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent

SOURCE = "11b575d0d8b3f33a9671fdf8708a070ef0a878ee4f9606666a6018ec96982a97"
COHORTS = {
    "casia5": dict(root="results_casia_validation_v2", split="validation", seeds=[11, 22, 33, 44, 55],
                   probes=792, identities=12,
                   methods=["ce", "masked_ce", "matched_ce", "metric_baseline",
                            "metric_global_supcon", "metric_part_supcon", "oap_v2"]),
    "gait3d_dev": dict(root="results_gait3d_v2_comparison", split="validation", seeds=[11, 22, 33],
                       probes=300, identities=300,
                       methods=["ce", "matched_ce", "metric_baseline", "oap_v2"]),
    "gait3d_test": dict(root="results_gait3d_v2_comparison_final", split="test", seeds=[11, 22, 33, 44, 55],
                        probes=947, identities=947, methods=["ce", "metric_baseline"]),
}
ENDPOINTS = ["clean", "robust", "occluded", "family:random_joint", "family:dynamic_joint",
             "family:body_part", "family:temporal"]
DESCRIPTORS = ["global", "raw_reliable", "raw_uniform"]


def pct(x):
    return None if x is None else 100.0 * float(x)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def cohort(name, spec):
    base = ROOT / spec["root"] / "comparison"
    meta = json.loads((base / "comparison.json").read_text())
    for key in ("seeds", "probes", "identities", "methods"):
        if meta[key] != spec[key]:
            raise SystemExit(f"{name}: {key} is {meta[key]}, expected {spec[key]}")
    if meta["evaluation_split"] != spec["split"] or meta["source_sha256"] != SOURCE:
        raise SystemExit(f"{name}: unexpected split or source revision")
    seed_columns = [f"seed{s}" for s in spec["seeds"]]
    summary = {}
    for row in read_csv(base / "comparison_summary.csv"):
        values = [float(row[c]) for c in seed_columns]
        cell = {"mean": pct(row["mean"]), "ci95": [pct(row["ci95_low"]), pct(row["ci95_high"])],
                "sd": float(np.std(np.multiply(values, 100), ddof=1)),
                "seeds": [pct(v) for v in values]}
        if abs(np.mean(values) * 100 - cell["mean"]) > 1e-6:
            raise SystemExit(f"{name}: seed columns disagree with the mean")
        summary.setdefault(row["method"], {}).setdefault(row["descriptor"], {}) \
            .setdefault(row["endpoint"], {})[row["metric"]] = cell
    contrasts = {}
    for row in read_csv(base / "comparison_contrasts.csv"):
        key = f"{row['method']}|{row['comparison']}" if row["type"] == "descriptor" else row["comparison"]
        contrasts.setdefault(row["type"], {}).setdefault(key, {}).setdefault(row["descriptor"], {}) \
            .setdefault(row["endpoint"], {})[row["metric"]] = {
                "mean": pct(row["mean"]), "ci95": [pct(row["ci95_low"]), pct(row["ci95_high"])],
                "p_positive": float(row["probability_positive"])}
    out = {"meta": {k: meta[k] for k in ("dataset", "evaluation_split", "seeds", "probes", "identities",
                                         "epochs", "source_sha256", "dataset_sha256", "bootstrap")},
           "summary": summary, "contrasts": contrasts,
           "inputs": {p.name: sha(p) for p in sorted(base.glob("*")) if p.is_file()}}
    official = base / "official_single_view.csv"
    if official.exists():
        rows = read_csv(official)
        out["single_view"] = {}
        for row in rows:
            out["single_view"].setdefault(row["method"], {}).setdefault(row["descriptor"], {})[
                row["condition"]] = {"mean": pct(row["rank1_mean"]), "sd": pct(row["rank1_sd"])}
        for method, by_descriptor in out["single_view"].items():
            for descriptor, conditions in by_descriptor.items():
                conditions["average"] = {"mean": float(np.mean([conditions[c]["mean"] for c in ("NM", "BG", "CL")]))}
    return out


def run_level(name, spec):
    """Per-run metadata and standard-protocol rows that compare_runs does not summarize."""
    root = ROOT / spec["root"]
    rows, hours = {}, {}
    for directory in sorted(root.glob("*/metrics.json")):
        metrics = json.loads(directory.read_text())
        if metrics.get("status") != "complete" or metrics["source_sha256"] != SOURCE:
            continue
        method, seed = metrics["method"], int(metrics["seed"])
        if method not in spec["methods"] or seed not in spec["seeds"]:
            continue
        hours.setdefault(method, []).append(float(metrics["train_hours"]))
        for row in metrics["results"]:
            if row["protocol"] == "clean_gallery" and row["corruption"] == "official_condition:in_the_wild":
                rows.setdefault(method, {}).setdefault(row["descriptor"], []).append(
                    {k: pct(row[k]) for k in ("rank1", "rank5", "rank10", "map", "minp")})
        rows.setdefault("_parameters", {})[method] = int(metrics["parameters"])
    for method in spec["methods"]:
        if len(hours.get(method, [])) != len(spec["seeds"]):
            raise SystemExit(f"{name}: {method} has {len(hours.get(method, []))} complete runs")
    standard = {}
    for method, by_descriptor in rows.items():
        if method == "_parameters":
            continue
        for descriptor, values in by_descriptor.items():
            standard.setdefault(method, {})[descriptor] = {
                metric: {"mean": float(np.mean([v[metric] for v in values])),
                         "sd": float(np.std([v[metric] for v in values], ddof=1))}
                for metric in values[0]}
    return {"train_hours": {m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for m, v in hours.items()},
            "parameters": rows.get("_parameters", {}), "standard_protocol": standard}


def controls(path: Path):
    analysis = json.loads(path.read_text())
    keep = {"design": analysis["design"], "acceptance_gate": analysis["acceptance_gate"],
            "random_head_stability": analysis["random_head_stability"],
            "recommendation": analysis["recommendation"], "sha256": sha(path)}
    keep["descriptors"] = {}
    for name in ("raw_reliable", "raw_uniform", "frozen_head_reliable"):
        keep["descriptors"][name] = {w: {k: pct(v) if isinstance(v, float) and k != "families_improved" else v
                                         for k, v in cell.items()}
                                     for w, cell in analysis["sensitivity"][name].items()}
    keep["global"] = {k: pct(v) for k, v in analysis["global"].items()}
    random_heads = [n for n in analysis["sensitivity"] if n.startswith("random_head_")]
    keep["random_heads_primary"] = {
        endpoint: {"mean": float(np.mean([pct(analysis["sensitivity"][n]["0.5"][endpoint]) for n in random_heads])),
                   "range": [float(np.min([pct(analysis["sensitivity"][n]["0.5"][endpoint]) for n in random_heads])),
                             float(np.max([pct(analysis["sensitivity"][n]["0.5"][endpoint]) for n in random_heads]))]}
        for endpoint in ("clean", "robust", "severity_05", "occluded")}
    keep["paired_bootstrap"] = {
        d: {e: {"difference": pct(v["difference"]), "ci95": [pct(v["ci95"][0]), pct(v["ci95"][1])],
                "p_positive": v["probability_positive"]} for e, v in cells.items()}
        for d, cells in analysis["paired_bootstrap"].items()}
    return keep


def controls_three_seed():
    """Means of the original three-seed checkpoint controls (intervals come from the IJCV reanalysis)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_descriptor_controls as module
    seeds = [11, 22, 33]
    runs, design = module.discover(ROOT / "results_descriptor_controls", seeds)
    by_seed = module.result_rows(runs, seeds)
    names = ["global", "frozen_head_reliable", "raw_uniform", "raw_reliable"] + \
        [f"random_head_{s}_reliable" for s in design["random_head_seeds"]]
    out = {}
    for name in names:
        weight = 0.0 if name == "global" else 0.5
        out[name] = {k: pct(v) for k, v in module.aggregate_endpoints(by_seed, seeds, name, weight).items()}
    heads = [n for n in names if n.startswith("random_head_")]
    out["random_heads_primary"] = {e: {"mean": float(np.mean([out[n][e] for n in heads])),
                                       "range": [float(min(out[n][e] for n in heads)), float(max(out[n][e] for n in heads))]}
                                   for e in ("clean", "robust", "severity_05", "occluded")}
    out["sources"] = {r.name: sha(r / "metrics.json") for r in runs.values()}
    return out


def casia_official_test():
    """Five-seed official-test runs (identities 075-124, standard single-view protocol)."""
    per = {}
    sources = {}
    for path in sorted((ROOT / "results_v2").glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        if metrics.get("evaluation_split") != "test" or metrics.get("status") != "complete":
            continue
        sources[path.parent.name] = {"sha256": sha(path), "source_sha256": metrics["source_sha256"],
                                     "config_fingerprint": metrics["config_fingerprint"], "epochs": metrics["epochs"],
                                     "parameters": int(metrics["parameters"])}
        for row in metrics["results"]:
            if row["protocol"] == "single_view_gallery" and row["corruption"].startswith("official_condition:") \
                    and float(row["severity"]) == 0:
                condition = row["corruption"].split(":")[1]
                per.setdefault(row["descriptor"], {}).setdefault(condition, {})[int(metrics["seed"])] = 100 * row["rank1"]
    if len(sources) != 5:
        raise SystemExit(f"expected five official-test runs, found {len(sources)}")
    parameters = {v["parameters"] for v in sources.values()}
    if len(parameters) != 1:
        raise SystemExit(f"official-test runs disagree on parameter count: {parameters}")
    out = {"runs": sources, "parameters_million": parameters.pop() / 1e6}
    for descriptor, conditions in per.items():
        seeds = sorted(conditions["NM"])
        average = [np.mean([conditions[c][s] for c in ("NM", "BG", "CL")]) for s in seeds]
        out[descriptor] = {c: {"mean": float(np.mean(list(v.values()))), "sd": float(np.std(list(v.values()), ddof=1))}
                           for c, v in conditions.items()}
        out[descriptor]["average"] = {"mean": float(np.mean(average)), "sd": float(np.std(average, ddof=1))}
        out[descriptor]["_average_per_seed"] = dict(zip(seeds, map(float, average)))
    # Paired t interval over the five seed-level differences (t(0.975, 4) = 2.776).
    a, g = out["raw_reliable"]["_average_per_seed"], out["global"]["_average_per_seed"]
    diffs = np.array([a[s] - g[s] for s in sorted(a)])
    half = 2.7764451051977987 * diffs.std(ddof=1) / np.sqrt(len(diffs))
    out["raw_reliable_minus_global_average"] = {"difference": float(diffs.mean()),
                                                "ci95": [float(diffs.mean() - half), float(diffs.mean() + half)],
                                                "per_seed": diffs.tolist()}
    return out


def fastposegait():
    """Parse the official-framework test logs and the framework's published table."""
    out = {}
    for log in sorted((ROOT / "logs").glob("fastposegait_*.out")):
        text = log.read_text(errors="ignore")
        name = re.search(r"'save_name': 'oap_(casia_b_pose|gait3d)_final_(\w+?)'", text)
        if not name or "out of memory" in text.lower():
            continue
        dataset, model = name.groups()
        params = re.search(r"Parameters Count: ([\d.]+)M", text)
        record = {"log": log.name, "log_sha256": sha(log), "parameters_million": float(params.group(1)) if params else None}
        casia = re.findall(r"NM@R1: ([\d.]+)%\s+BG@R1: ([\d.]+)%\s+CL@R1: ([\d.]+)%", text)
        gait3d = re.findall(r"'scalar/test_accuracy/Rank-1': np\.float32\(([\d.]+)\), "
                            r"'scalar/test_accuracy/Rank-5': np\.float32\(([\d.]+)\), "
                            r"'scalar/test_accuracy/Rank-10': np\.float32\(([\d.]+)\), "
                            r"'scalar/test_accuracy/mAP': np\.float64\(([\d.]+)\)"
                            r"(?:, 'scalar/test_accuracy/mINP': np\.float64\(([\d.]+)\))?", text)
        if casia:
            nm, bg, cl = map(float, casia[-1])
            record.update(NM=nm, BG=bg, CL=cl, average=(nm + bg + cl) / 3)
        elif gait3d:
            values = [float(v) if v else None for v in gait3d[-1]]
            record.update(dict(zip(("rank1", "rank5", "rank10", "map", "minp"), values)))
        else:
            continue
        out.setdefault(dataset, {})[model] = record  # later logs supersede earlier ones
    zoo = (ROOT / "external/FastPoseGait/docs/model_zoo.md").read_text().split("## Improved Version")[0]
    published = {"source": "external/FastPoseGait/docs/model_zoo.md (vanilla version)"}
    casia_section, rest = zoo.split("###  [OUMVLP-Pose]")[0], zoo.split("###  [Gait3D]")[1]
    for model, key in (("GaitGraph2", "gaitgraph2"), ("GPGait", "gpgait")):
        row = next(line for line in casia_section.splitlines() if f"[{model}]" in line)
        cells = [c.strip() for c in row.split("|") if c.strip()]
        nums = [float(re.match(r"[\d.]+", c).group()) for c in cells[-4:]]
        published.setdefault("casia_b_pose", {})[key] = dict(zip(("NM", "BG", "CL", "average"), nums))
        row = next(line for line in rest.splitlines() if f"[{model}]" in line)
        published.setdefault("gait3d", {})[key] = {"rank1": float([c.strip() for c in row.split("|") if c.strip()][-1])}
    out["published"] = published
    return out


def gpgait_transfer():
    """Raw part matching on the reproduced GPGait checkpoints (scripts/fpg_part_matching.py)."""
    out = {}
    for dataset in ("casia_b", "gait3d"):
        for variant, suffix in (("primary", ""), ("projected", "_projected")):
            path = ROOT / f"results_fpg_part_matching/gpgait_{dataset}{suffix}.json"
            data = json.loads(path.read_text())
            data["_sha256"] = sha(path)
            out.setdefault(dataset, {})[variant] = data
    # The extraction must reproduce GPGait's logged test accuracy before any derived result is used.
    logged = fastposegait()
    full_casia = out["casia_b"]["primary"]["full"]
    for condition in ("NM", "BG", "CL"):
        if abs(full_casia[condition] - logged["casia_b_pose"]["gpgait"][condition]) > 0.1:
            raise SystemExit(f"GPGait CASIA-B {condition} not reproduced: {full_casia[condition]}")
    if abs(out["gait3d"]["primary"]["full"]["rank1"] - logged["gait3d"]["gpgait"]["rank1"]) > 0.1:
        raise SystemExit("GPGait Gait3D Rank-1 not reproduced")
    return out


def efficiency():
    out = {}
    for log in sorted((ROOT / "logs").glob("oap_efficiency_*.out")):
        text = log.read_text()
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        out[log.name] = data
    return out


def epoch_budget():
    out = {}
    for method in ("metric_baseline", "ce"):
        runs = sorted(str(p) for p in (ROOT / "results_gait3d_v2_comparison").glob(f"gait3d_multistream_{method}_seed*"))
        result = subprocess.run([sys.executable, str(ROOT / "scripts/choose_epoch_budget.py"), *runs],
                                capture_output=True, text=True, check=True, cwd=ROOT,
                                env={"PYTHONPATH": str(ROOT / "src")})
        decision = json.loads(result.stdout)
        out[method] = {k: decision[k] for k in ("decision", "budget", "reason", "per_seed_best_epoch",
                                                "last_minus_peak", "schedule_epochs")}
    return out


def missingness():
    out = {}
    for name, root in (("casia_b_pose", "results_casia_validation_v2"), ("gait3d", "results_gait3d_v2_comparison")):
        data = json.loads((ROOT / root / "actual_missingness/actual_missingness.json").read_text())
        out[name] = {k: {"mean": 100 * v["mean"], "min": 100 * v["min"], "max": 100 * v["max"]}
                     for k, v in data["removed_fraction_by_nominal_severity"].items()}
    return out


def severity_curves():
    """Per-seed clean-gallery Rank-1 for every family and nominal severity (CASIA-B, five seeds)."""
    spec = COHORTS["casia5"]
    lines = [("CE, no masking / global", "ce", "global", "#8c8c8c", "v", ":"),
             ("Matched CE / global", "matched_ce", "global", "#ad761a", "s", "--"),
             ("Metric baseline / global", "metric_baseline", "global", "#264b75", "o", "-"),
             ("Part SupCon / global", "metric_part_supcon", "global", "#b34c46", "^", "-."),
             ("Metric baseline / raw parts", "metric_baseline", "raw_reliable", "#258366", "D", "-")]
    families = ["random_joint", "dynamic_joint", "body_part", "temporal"]
    severities = [0.0, 0.1, 0.3, 0.5, 0.7]
    cells = {}
    for path in sorted((ROOT / spec["root"]).glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        if metrics.get("status") != "complete":
            continue
        for row in metrics["results"]:
            if row["protocol"] == "clean_gallery" and row["corruption"] in families \
                    and float(row.get("fusion_weight") or 0) in (0, .5):
                key = (metrics["method"], row["descriptor"], row["corruption"], float(row["severity"]))
                cells.setdefault(key, {})[int(metrics["seed"])] = 100 * float(row["rank1"])
    curves = {}
    for label, method, descriptor, *_ in lines:
        for family in families:
            means, sds = [], []
            for severity in severities:
                values = cells.get((method, descriptor, family, severity), {})
                if sorted(values) != spec["seeds"]:
                    raise SystemExit(f"curve cell {method}/{descriptor}/{family}/{severity}: seeds {sorted(values)}")
                means.append(float(np.mean(list(values.values()))))
                sds.append(float(np.std(list(values.values()), ddof=1)))
            curves.setdefault(label, {})[family] = {"mean": means, "sd": sds}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), sharex=True, sharey=True)
    titles = ["Random joint-frame", "Dynamic joint (held out from training)", "Whole part", "Temporal interruption"]
    for ax, family, title in zip(axes.flat, families, titles):
        for label, method, descriptor, color, marker, style in lines:
            mean = np.array(curves[label][family]["mean"])
            sd = np.array(curves[label][family]["sd"])
            ax.plot(severities, mean, label=label, color=color, marker=marker, linestyle=style,
                    linewidth=1.4, markersize=3.5)
            ax.fill_between(severities, mean - sd, mean + sd, color=color, alpha=.10, linewidth=0)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks(severities)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linewidth=.4, alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Nominal corruption severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("Rank-1 (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(HERE / "fig4.pdf", bbox_inches="tight")
    if os.environ.get("PR_PREVIEW"):
        fig.savefig(Path(os.environ["PR_PREVIEW"]) / "fig4_severity.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return curves


def forest(evidence):
    """Raw reliability-weighted part matching minus global matching, every cohort and objective."""
    names = {"ce": "CE, no masking", "masked_ce": "Masked CE", "matched_ce": "Matched CE",
             "metric_baseline": "Metric baseline", "metric_global_supcon": "Global SupCon",
             "metric_part_supcon": "Part SupCon", "oap_v2": "Global + part SupCon"}
    rows = []
    factorial = evidence["factorial_3seed"]["descriptor_effects"]["raw_reliable"]
    rows.append(("CASIA-B dev., 3 seeds", "Metric baseline", factorial["clean"], factorial["robust"]))
    for cohort_name, label in (("casia5", "CASIA-B dev., 5 seeds"), ("gait3d_dev", "Gait3D dev., 3 seeds"),
                               ("gait3d_test", "Gait3D test, 5 seeds")):
        contrasts = evidence[cohort_name]["contrasts"]["descriptor"]
        for method in COHORTS[cohort_name]["methods"]:
            cell = contrasts[f"{method}|raw_reliable_minus_global"]["raw_reliable"]
            convert = lambda c: {"difference": c["rank1"]["mean"], "ci95": c["rank1"]["ci95"]}
            rows.append((label, names[method], convert(cell["clean"]), convert(cell["robust"])))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.24 * len(rows) + 0.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    colors = {"CASIA-B dev., 3 seeds": "#7a7a7a", "CASIA-B dev., 5 seeds": "#264b75",
              "Gait3D dev., 3 seeds": "#ad761a", "Gait3D test, 5 seeds": "#258366"}
    for ax, index, title in ((axes[0], 2, "Uncorrupted probes"),
                             (axes[1], 3, "Corrupted probes, clean gallery")):
        for yy, row in zip(y, rows):
            cell = row[index]
            low, high = cell["ci95"]
            ax.plot([low, high], [yy, yy], color=colors[row[0]], linewidth=1.4)
            ax.plot(cell["difference"], yy, "o", color=colors[row[0]], markersize=3.8)
        ax.axvline(0, color="black", linewidth=.7)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", linewidth=.4, alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r[0]}: {r[1]}" for r in rows], fontsize=7.5)
    fig.supxlabel("Rank-1 difference, raw parts minus global descriptor (percentage points; 95% interval)",
                  fontsize=8.5)
    fig.tight_layout()
    fig.savefig(HERE / "fig3.pdf", bbox_inches="tight")
    if os.environ.get("PR_PREVIEW"):
        fig.savefig(Path(os.environ["PR_PREVIEW"]) / "fig3_forest.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return [{"cohort": r[0], "objective": r[1], "clean": r[2], "robust": r[3]} for r in rows]


def summary_figure(evidence):
    """Figure 1: test-set Rank-1 change from regional evidence, our encoder versus GPGait."""
    t, g = evidence["gpgait_transfer"]["casia_b"]["primary"], evidence["gpgait_transfer"]["gait3d"]["primary"]
    ours_casia = evidence["casia_official_test"]["raw_reliable_minus_global_average"]
    ours_gait3d = evidence["gait3d_test"]["contrasts"]["descriptor"]["metric_baseline|raw_reliable_minus_global"]["raw_reliable"]["clean"]["rank1"]
    cell = lambda c: {"difference": c["difference"], "ci95": c["ci95"]}
    rows = [("Our encoder\n+ raw parts", cell(ours_casia), {"difference": ours_gait3d["mean"], "ci95": ours_gait3d["ci95"]}),
            ("GPGait\n+ raw parts", cell(t["_contrasts_average"]["full+raw - full"]), cell(g["_contrasts_rank1"]["full+raw - full"])),
            ("GPGait's trained\npart branches", {"difference": -t["_contrasts_average"]["global - full"]["difference"],
                                                 "ci95": [-v for v in t["_contrasts_average"]["global - full"]["ci95"][::-1]]},
             {"difference": -g["_contrasts_rank1"]["global - full"]["difference"],
              "ci95": [-v for v in g["_contrasts_rank1"]["global - full"]["ci95"][::-1]]})]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    colors = {"CASIA-B": "#264b75", "Gait3D": "#258366"}
    for k, (label, casia, gait3d) in enumerate(rows):
        for offset, (name, c) in ((0.16, ("CASIA-B", casia)), (-0.16, ("Gait3D", gait3d))):
            y = len(rows) - 1 - k + offset
            ax.plot(c["ci95"], [y, y], color=colors[name], linewidth=1.6)
            ax.plot(c["difference"], y, "o", color=colors[name], markersize=4,
                    label=name if k == 0 else None)
    ax.axvline(0, color="black", linewidth=.7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlabel("Change in test-set Rank-1 (points; 95% interval)")
    ax.grid(axis="x", linewidth=.4, alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=7, handletextpad=.2, borderaxespad=.1)
    fig.tight_layout()
    fig.savefig(HERE / "fig1.pdf", bbox_inches="tight")
    if os.environ.get("PR_PREVIEW"):
        fig.savefig(Path(os.environ["PR_PREVIEW"]) / "fig1_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [{"label": r[0], "casia_b": r[1], "gait3d": r[2]} for r in rows]


def main():
    evidence = {"source_sha256": SOURCE}
    factorial_path = ROOT / "Paper/IJCV/evidence_crossed.json"
    factorial = json.loads(factorial_path.read_text())
    scaled = lambda c: {"difference": 100 * c["difference"], "ci95": [100 * v for v in c["ci95"]]}
    evidence["factorial_3seed"] = {
        "sha256": sha(factorial_path),
        "against_baseline": {k: {e: scaled(c) for e, c in v.items()} for k, v in factorial["against_baseline"].items()},
        "descriptor_effects": {k: {e: scaled(c) for e, c in v.items()} for k, v in factorial["descriptor_effects"].items()},
        "bootstrap": factorial["bootstrap"],
    }
    for name, spec in COHORTS.items():
        evidence[name] = cohort(name, spec)
        evidence[name]["runs"] = run_level(name, spec)
    evidence["casia_official_test"] = casia_official_test()
    evidence["controls_casia3"] = controls_three_seed()
    evidence["controls_casia5"] = controls(ROOT / "results_descriptor_controls_5seed/analysis.json")
    evidence["controls_gait3d_dev"] = controls(ROOT / "results_descriptor_controls_gait3d/analysis.json")
    occlusion = json.loads((ROOT / "results_gait3d_v2_comparison/occlusion_strata/occlusion_analysis.json").read_text())
    evidence["gait3d_occlusion"] = {k: occlusion[k] for k in ("evaluation_split", "seeds", "probes", "primary",
                                                               "secondary", "bin_edges", "confounder_spearman",
                                                               "bootstrap_samples")}
    evidence["missingness"] = missingness()
    evidence["fastposegait"] = fastposegait()
    evidence["efficiency"] = efficiency()
    evidence["gpgait_transfer"] = gpgait_transfer()
    evidence["epoch_budget_gait3d"] = epoch_budget()
    evidence["curves_casia5"] = severity_curves()
    evidence["forest"] = forest(evidence)
    evidence["summary_figure"] = summary_figure(evidence)
    (HERE / "evidence_pr.json").write_text(json.dumps(evidence, indent=1) + "\n")
    import make_tables
    make_tables.write_all(evidence, HERE / "tables")
    # The T-BIOM version uses full-width IEEE tables, so intervals sit on one line.
    if (HERE.parent / "TBIOM").is_dir():
        make_tables.write_all(evidence, HERE.parent / "TBIOM" / "tables", stacked=False)
    print("wrote", HERE / "evidence_pr.json")


if __name__ == "__main__":
    main()
