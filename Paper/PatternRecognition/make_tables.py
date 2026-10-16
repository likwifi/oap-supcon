"""LaTeX table bodies for the Pattern Recognition manuscript, generated from evidence_pr.json.

Called by build_pr_evidence.py. Every number is formatted here from the evidence
dictionary; captions and surrounding text live in main.tex / supplementary.tex.
"""
from __future__ import annotations

from pathlib import Path

NAMES = {
    "ce": "CE, no masking",
    "masked_ce": "Masked CE",
    "matched_ce": "Matched CE",
    "metric_baseline": "Metric baseline",
    "metric_global_supcon": "Global SupCon",
    "metric_part_supcon": "Part SupCon",
    "metric_part_triplet": "Part triplet",
    "oap_v2": "Global + part SupCon",
    "oap_v2_ungated": "Global + part SupCon, ungated",
}
CASIA_ORDER = ["ce", "masked_ce", "matched_ce", "metric_baseline", "metric_global_supcon",
               "metric_part_supcon", "oap_v2"]


def n(x, digits=2):
    """Number in math mode with a true minus sign."""
    return f"${x:.{digits}f}$"


def signed(x, digits=2):
    return f"${x:+.{digits}f}$".replace("+", "{+}") if x >= 0 else f"${x:.{digits}f}$"


def ci(cell, digits=2):
    low, high = cell
    return f"[{low:.{digits}f}, {high:.{digits}f}]".replace("-", "$-$")


def diff(mean, interval, digits=2):
    return f"{signed(mean, digits)} {ci(interval, digits)}"


STACKED = True


def stack(mean, interval, digits=2):
    """Estimate above its interval for narrow columns; one line when STACKED is False."""
    if not STACKED:
        return diff(mean, interval, digits)
    # \relax stops LaTeX from reading the interval's "[" as a \\ spacing argument.
    return (r"\begin{tabular}[c]{@{}c@{}}" + signed(mean, digits) + r"\\\relax " +
            ci(interval, digits) + r"\end{tabular}")


def head(*lines):
    return r"\begin{tabular}[b]{@{}c@{}}" + r"\\".join(lines) + r"\end{tabular}"


def pm(mean, sd, digits=2):
    return f"${mean:.{digits}f}\\pm{sd:.{digits}f}$"


def table_casia5(e):
    s = e["casia5"]["summary"]
    sv = e["casia5"]["single_view"]
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Global descriptor} & \multicolumn{2}{c}{Raw parts} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-7}",
             "Training objective & Clean & " + head("Corr.", "probe") + " & " + head("Corr.", "gallery") + " & "
             + head("Single-", "view") + " & Clean & " + head("Corr.", "probe") + r" \\",
             r"\midrule"]
    for m in CASIA_ORDER:
        g, r = s[m]["global"], s[m]["raw_reliable"]
        lines.append(" & ".join([
            NAMES[m],
            pm(g["clean"]["rank1"]["mean"], g["clean"]["rank1"]["sd"]),
            pm(g["robust"]["rank1"]["mean"], g["robust"]["rank1"]["sd"]),
            n(g["occluded"]["rank1"]["mean"]),
            n(sv[m]["global"]["average"]["mean"]),
            n(r["clean"]["rank1"]["mean"]),
            n(r["robust"]["rank1"]["mean"]),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_paired(e):
    c5 = e["casia5"]["contrasts"]["method"]
    gd = e["gait3d_dev"]["contrasts"]["method"]
    f3 = e["factorial_3seed"]["against_baseline"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"\multicolumn{4}{@{}l}{\emph{(a) CASIA-B development}} \\",
             r" & \multicolumn{2}{c}{Five seeds} & Three seeds \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-4}",
             r"Training objective & Clean & Corrupted probe & Corrupted probe \\", r"\midrule"]
    for m in ["ce", "masked_ce", "matched_ce", "metric_global_supcon", "metric_part_supcon",
              "metric_part_triplet", "oap_v2", "oap_v2_ungated"]:
        key, fk = f"{m}_minus_metric_baseline", f"{m}/global"
        cells = [NAMES[m]]
        if key in c5:
            g = c5[key]["global"]
            cells += [stack(g["clean"]["rank1"]["mean"], g["clean"]["rank1"]["ci95"]),
                      stack(g["robust"]["rank1"]["mean"], g["robust"]["rank1"]["ci95"])]
        else:
            cells += ["--", "--"]
        cells.append(stack(f3[fk]["robust"]["difference"], f3[fk]["robust"]["ci95"]) if fk in f3 else "--")
        lines.append(" & ".join(cells) + r" \\[3pt]")
    lines += [r"\midrule", r"\multicolumn{4}{@{}l}{\emph{(b) Gait3D development, three seeds}} \\",
              r"Training objective & Clean & Corrupted probe & Corrupted gallery \\", r"\midrule"]
    for m in ["ce", "matched_ce", "oap_v2"]:
        g = gd[f"{m}_minus_metric_baseline"]["global"]
        lines.append(" & ".join([NAMES[m]] + [stack(g[ep]["rank1"]["mean"], g[ep]["rank1"]["ci95"])
                                              for ep in ("clean", "robust", "occluded")]) + r" \\[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_controls(e):
    c3, c5, gd = e["controls_casia3"], e["controls_casia5"], e["controls_gait3d_dev"]

    def value(c, name, endpoint):
        if name == "global":
            return c["global"][endpoint]
        if name == "random":
            return c["random_heads_primary"][endpoint]["mean"]
        if "descriptors" in c:
            return c["descriptors"][name]["0.5"][endpoint]
        return c[name][endpoint]

    rows = [("Global only", "global"), ("Saved untrained head", "frozen_head_reliable"),
            ("Random heads (mean of 10)", "random"), ("Raw parts, uniform", "raw_uniform"),
            ("Raw parts, reliability", "raw_reliable")]
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"\multicolumn{7}{@{}l}{\emph{(a) Rank-1 (\%)}} \\",
             r" & \multicolumn{2}{c}{CASIA-B, 3 seeds} & \multicolumn{2}{c}{CASIA-B, 5 seeds} & \multicolumn{2}{c}{Gait3D, 3 seeds} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             "Descriptor" + (" & Clean & " + head("Corr.", "probe")) * 3 + r" \\",
             r"\midrule"]
    for label, name in rows:
        cells = [label]
        for c in (c3, c5, gd):
            cells += [n(value(c, name, "clean")), n(value(c, name, "robust"))]
        lines.append(" & ".join(cells) + r" \\[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{6pt}", "",
              r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
              r"\multicolumn{5}{@{}l}{\emph{(b) Paired differences (points)}} \\",
              r"Contrast & Endpoint & CASIA-B, 3 seeds & CASIA-B, 5 seeds & Gait3D, 3 seeds \\", r"\midrule"]
    f3 = e["factorial_3seed"]["descriptor_effects"]
    for label, key3, key in (("Reliability $-$ global", "raw_reliable", "raw_reliable"),
                             ("Reliability $-$ uniform", "raw_reliable_minus_uniform", "raw_reliable_minus_uniform")):
        for ep, ep_label in (("clean", "Clean"), ("robust", "Corr.\\ probe")):
            cells = [label if ep == "clean" else "", ep_label,
                     stack(f3[key3][ep]["difference"], f3[key3][ep]["ci95"])]
            cells += [stack(c["paired_bootstrap"][key][ep]["difference"], c["paired_bootstrap"][key][ep]["ci95"])
                      for c in (c5, gd)]
            lines.append(" & ".join(cells) + r" \\[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_casia_test(e):
    t = e["casia_official_test"]
    fpg = e["fastposegait"]
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"System & NM & BG & CL & Average & " + head("Params.", "(M)") + r" \\", r"\midrule"]
    params = e["casia_official_test"].get("parameters_million")
    for label, d in (("Ours, global", "global"), ("Ours, raw (uniform)", "raw_uniform"),
                     ("Ours, raw (reliability)", "raw_reliable")):
        lines.append(" & ".join([label] + [pm(t[d][c]["mean"], t[d][c]["sd"]) for c in ("NM", "BG", "CL", "average")]
                                + [n(params, 2) if params else "--"]) + r" \\")
    lines.append(r"\midrule")
    for model, label in (("gaitgraph2", "GaitGraph2"), ("gpgait", "GPGait")):
        r = fpg["casia_b_pose"][model]
        lines.append(" & ".join([f"{label} (reproduced)"] + [n(r[c]) for c in ("NM", "BG", "CL", "average")]
                                + [n(r["parameters_million"], 2)]) + r" \\")
        p = fpg["published"]["casia_b_pose"][model]
        lines.append(" & ".join([f"{label} (reported)"]
                                + [n(p[c]) for c in ("NM", "BG", "CL", "average")] + ["--"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_gait3d_test(e):
    sp = e["gait3d_test"]["runs"]["standard_protocol"]
    params = e["gait3d_test"]["runs"]["parameters"]
    fpg = e["fastposegait"]
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"System & Rank-1 & Rank-5 & Rank-10 & mAP & mINP & " + head("Params.", "(M)") + r" \\", r"\midrule"]
    for method, d, label in (("ce", "global", "CE, global"),
                             ("ce", "raw_reliable", "CE, raw (reliability)"),
                             ("metric_baseline", "global", "Metric, global"),
                             ("metric_baseline", "raw_uniform", "Metric, raw (uniform)"),
                             ("metric_baseline", "raw_reliable", "Metric, raw (reliability)")):
        c = sp[method][d]
        lines.append(" & ".join([label, pm(c["rank1"]["mean"], c["rank1"]["sd"]), n(c["rank5"]["mean"]),
                                 n(c["rank10"]["mean"]), pm(c["map"]["mean"], c["map"]["sd"]), n(c["minp"]["mean"]),
                                 n(params[method] / 1e6)]) + r" \\")
    lines.append(r"\midrule")
    for model, label in (("gaitgraph2", "GaitGraph2"), ("gpgait", "GPGait")):
        r = fpg["gait3d"][model]
        lines.append(" & ".join([f"{label} (reproduced)", n(r["rank1"]), n(r["rank5"]), n(r["rank10"]),
                                 n(r["map"]), n(r["minp"]), n(r["parameters_million"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_main_comparison(e):
    """One test-set table: our encoder, reproduced and reported baselines, and GPGait descriptor variants."""
    t, fpg = e["casia_official_test"], e["fastposegait"]
    sp, params = e["gait3d_test"]["runs"]["standard_protocol"]["metric_baseline"], e["gait3d_test"]["runs"]["parameters"]
    gp_c, gp_g = e["gpgait_transfer"]["casia_b"]["primary"], e["gpgait_transfer"]["gait3d"]["primary"]
    casia_params = t["parameters_million"]
    lines = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{CASIA-B official test} & \multicolumn{2}{c}{Gait3D test} & \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
             r"System & NM & BG & CL & Avg. & Rank-1 & mAP & " + head("Params. (M)", "CASIA / G3D") + r" \\",
             r"\midrule", r"\multicolumn{8}{@{}l}{\emph{(a) Systems}} \\"]
    for label, d, g3 in (("Ours, global (5 runs)", "global", "global"),
                         ("Ours, raw parts (5 runs)", "raw_reliable", "raw_reliable")):
        lines.append(" & ".join([label] + [n(t[d][c]["mean"]) for c in ("NM", "BG", "CL", "average")]
                                + [n(sp[g3]["rank1"]["mean"]), n(sp[g3]["map"]["mean"]),
                                   f"{casia_params:.2f} / {params['metric_baseline'] / 1e6:.2f}"]) + r" \\")
    for model, label in (("gaitgraph2", "GaitGraph2"), ("gpgait", "GPGait")):
        c, g = fpg["casia_b_pose"][model], fpg["gait3d"][model]
        lines.append(" & ".join([f"{label}, reproduced"] + [n(c[k]) for k in ("NM", "BG", "CL", "average")]
                                + [n(g["rank1"]), n(g["map"]),
                                   f"{c['parameters_million']:.2f} / {g['parameters_million']:.2f}"]) + r" \\")
        pc, pg = fpg["published"]["casia_b_pose"][model], fpg["published"]["gait3d"][model]
        lines.append(" & ".join([f"{label}, reported"] + [n(pc[k]) for k in ("NM", "BG", "CL", "average")]
                                + [n(pg["rank1"]), "--", "--"]) + r" \\")
    lines += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{(b) GPGait descriptor variants (same checkpoints)}} \\"]
    for key, label in (("global", "Whole-body only"), ("global+raw", "Whole-body + raw parts"),
                       ("full+raw", "Full + raw parts")):
        lines.append(" & ".join([label] + [n(gp_c[key][k]) for k in ("NM", "BG", "CL", "average")]
                                + [n(gp_g[key]["rank1"]), n(gp_g[key]["map"]), ""]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_gpgait(e):
    t = e["gpgait_transfer"]
    c, g = t["casia_b"]["primary"], t["gait3d"]["primary"]
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{CASIA-B official test} & \multicolumn{2}{c}{Gait3D test} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-7}",
             r"GPGait descriptor & NM & BG & CL & Average & Rank-1 & mAP \\", r"\midrule"]
    for key, label in (("full", "Full (19 embeddings)"), ("global", "Whole-body only (W)"),
                       ("raw", "Raw parts only"), ("global+raw", "W + raw parts"),
                       ("full+raw", "Full + raw parts")):
        lines.append(" & ".join([label] + [n(c[key][k]) for k in ("NM", "BG", "CL", "average")]
                                + [n(g[key]["rank1"]), n(g[key]["map"])]) + r" \\")
    lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{Paired differences (points), 95\% interval over test identities}} \\"]
    for contrast, label in (("global - full", "W $-$ full"), ("full+raw - full", "Full + raw $-$ full"),
                            ("global+raw - global", "W + raw $-$ W")):
        a = c["_contrasts_average"][contrast]
        r1, mp = g["_contrasts_rank1"][contrast], g["_contrasts_map"][contrast]
        lines.append(label + r" & \multicolumn{4}{c}{" + diff(a["difference"], a["ci95"]) + "} & "
                     + stack(r1["difference"], r1["ci95"]) + " & " + stack(mp["difference"], mp["ci95"]) + r" \\[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_gpgait_supplement(e):
    t = e["gpgait_transfer"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Descriptor & CASIA-B average & Gait3D Rank-1 & Gait3D mAP \\"]
    for variant, vlabel in (("primary", "Declared: mean pool, no projection"),
                            ("projected", "Exploratory: mean+max pool through the whole-body head")):
        c, g = t["casia_b"][variant], t["gait3d"][variant]
        lines += [r"\midrule", r"\multicolumn{4}{@{}l}{\emph{" + vlabel + r"}} \\"]
        for key, label in (("raw", "Raw parts only"), ("global+raw", "Whole-body + raw"), ("full+raw", "Full + raw"),
                           ("global+raw@0.25", r"Whole-body + raw, $\lambda=0.25$"),
                           ("full+raw@0.25", r"Full + raw, $\lambda=0.25$"),
                           ("global+raw@0.75", r"Whole-body + raw, $\lambda=0.75$"),
                           ("full+raw@0.75", r"Full + raw, $\lambda=0.75$")):
            lines.append(" & ".join([label, n(c[key]["average"]), n(g[key]["rank1"]), n(g[key]["map"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_weights(e):
    c5, gd = e["controls_casia5"], e["controls_gait3d_dev"]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{CASIA-B, five seeds} & \multicolumn{2}{c}{Gait3D, three seeds} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Local-score weight $\lambda$ & Clean & Corr.\ probe & Clean & Corr.\ probe \\", r"\midrule"]
    for w in ("0.0", "0.25", "0.5", "0.75", "1.0"):
        cells = [w + (" (declared)" if w == "0.5" else "")]
        for c in (c5, gd):
            cell = c["descriptors"]["raw_reliable"][w]
            cells += [n(cell["clean"]), n(cell["robust"])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_missingness(e):
    m = e["missingness"]
    families = [("random_joint", "Random joint-frame"), ("dynamic_joint", "Dynamic joint"),
                ("body_part", "Whole part"), ("temporal", "Temporal interruption")]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Family & $\rho=0.1$ & $\rho=0.3$ & $\rho=0.5$ & $\rho=0.7$ \\", r"\midrule"]
    for key, label in families:
        cells = [label]
        for sev in ("0.1", "0.3", "0.5", "0.7"):
            a, b = m["casia_b_pose"][f"{key}:{sev}"], m["gait3d"][f"{key}:{sev}"]
            cells.append(f"{a['mean']:.1f} / {b['mean']:.1f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_single_view(e):
    sv = e["casia5"]["single_view"]
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Global descriptor} & \multicolumn{3}{c}{Raw parts, reliability-weighted} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
             r"Training objective & NM & BG & CL & NM & BG & CL \\", r"\midrule"]
    for m in CASIA_ORDER:
        cells = [NAMES[m]]
        for d in ("global", "raw_reliable"):
            cells += [pm(sv[m][d][c]["mean"], sv[m][d][c]["sd"]) for c in ("NM", "BG", "CL")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_gait3d_dev(e):
    s = e["gait3d_dev"]["summary"]
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Global descriptor} & \multicolumn{2}{c}{Raw parts, reliability-weighted} \\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
             r"Training objective & Clean & Corr.\ probe & Corr.\ gallery & Clean & Corr.\ probe \\", r"\midrule"]
    for m in ("ce", "matched_ce", "metric_baseline", "oap_v2"):
        g, r = s[m]["global"], s[m]["raw_reliable"]
        lines.append(" & ".join([NAMES[m], pm(g["clean"]["rank1"]["mean"], g["clean"]["rank1"]["sd"]),
                                 pm(g["robust"]["rank1"]["mean"], g["robust"]["rank1"]["sd"]),
                                 n(g["occluded"]["rank1"]["mean"]), n(r["clean"]["rank1"]["mean"]),
                                 n(r["robust"]["rank1"]["mean"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_raw_by_objective(e):
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Cohort & Training objective & Clean & Corr.\ probe & Corr.\ gallery \\", r"\midrule"]
    for cohort, label, methods in (("casia5", "CASIA-B, 5 seeds", CASIA_ORDER),
                                   ("gait3d_dev", "Gait3D dev., 3 seeds", ["ce", "matched_ce", "metric_baseline", "oap_v2"]),
                                   ("gait3d_test", "Gait3D test, 5 seeds", ["ce", "metric_baseline"])):
        contrasts = e[cohort]["contrasts"]["descriptor"]
        for m in methods:
            c = contrasts[f"{m}|raw_reliable_minus_global"]["raw_reliable"]
            lines.append(" & ".join([label, NAMES[m]] + [diff(c[ep]["rank1"]["mean"], c[ep]["rank1"]["ci95"])
                                                       for ep in ("clean", "robust", "occluded")]) + r" \\")
        lines.append(r"\midrule" if cohort != "gait3d_test" else "")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(line for line in lines if line)


def write_all(e, directory: Path, stacked: bool = True):
    global STACKED
    STACKED = stacked
    directory.mkdir(exist_ok=True)
    tables = {
        "tab_casia5": table_casia5, "tab_paired": table_paired, "tab_controls": table_controls,
        "tab_casia_test": table_casia_test, "tab_gait3d_test": table_gait3d_test,
        "tab_s_weights": table_weights, "tab_s_missingness": table_missingness,
        "tab_s_single_view": table_single_view, "tab_s_gait3d_dev": table_gait3d_dev,
        "tab_s_raw_by_objective": table_raw_by_objective,
        "tab_gpgait": table_gpgait, "tab_s_gpgait": table_gpgait_supplement,
        "tab_main": table_main_comparison,
    }
    for name, function in tables.items():
        (directory / f"{name}.tex").write_text(
            f"% Generated by make_tables.py from evidence_pr.json; do not edit by hand.\n{function(e)}\n")
