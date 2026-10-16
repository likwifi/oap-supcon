#!/usr/bin/env python3
"""Generate the T-BIOM manuscript and supplement from the Pattern Recognition sources.

The PR main.tex and supplementary.tex are the single source of the text. This
script copies figures, converts the elsarticle front and back matter to
IEEEtran, makes tables and figures two-column floats, and applies the T-BIOM
layout changes listed in TBIOM_EDITS and SUPPLEMENT_EDITS. Every edit must
match exactly once, so a change in the PR text fails loudly instead of silently
producing a stale T-BIOM version. Tables in tables/ are written by
../PatternRecognition/build_pr_evidence.py (one-line intervals).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PR = HERE.parent / "PatternRecognition"

PREAMBLE = r"""\documentclass[journal]{IEEEtran}

\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\emergencystretch=2em
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dbltopfraction}{0.9}
\renewcommand{\dblfloatpagefraction}{0.85}
\setcounter{topnumber}{3}
\setcounter{dbltopnumber}{2}
\setcounter{totalnumber}{4}

\newcommand{\R}{\mathbb{R}}
\newcommand{\Lbase}{\mathcal{L}_{\mathrm{base}}}
\newcommand{\Lglobal}{\mathcal{L}_{\mathrm{gSC}}}
\newcommand{\Lpart}{\mathcal{L}_{\mathrm{pSC}}}
\newcommand{\Lptrip}{\mathcal{L}_{\mathrm{pTri}}}
\newcommand{\tablesetup}{\footnotesize\setlength{\tabcolsep}{4pt}}

\begin{document}

\title{Part Supervision Versus Part Matching for Pose-Based Gait Recognition Under Missing Joints}

\author{Minas~Aslanyan and Levon~Aslanyan%
\thanks{M. Aslanyan and L. Aslanyan are with the Institute for Informatics and
Automation Problems, National Academy of Sciences of the Republic of Armenia,
Yerevan, Armenia (e-mail: minas.aslanyan@iiap.sci.am; lasl@sci.am).
Corresponding author: Minas Aslanyan.}%
\thanks{This research did not receive any specific grant from funding agencies
in the public, commercial, or not-for-profit sectors.}}

\markboth{IEEE Transactions on Biometrics, Behavior, and Identity Science}%
{Aslanyan and Aslanyan: Part Supervision Versus Part Matching for Pose-Based Gait Recognition}

\maketitle

"""

BACK = r"""
\section*{Ethics Statement}
This study analyses only publicly released benchmark data, CASIA-B and Gait3D,
which were collected and distributed by their creators under their own
protocols and licences. No new data involving human participants were collected
and no participant was contacted.

\section*{Data and Code Availability}
CASIA-B poses are the HRNet-derived release distributed through FastPoseGait
(\url{https://github.com/BNU-IVC/FastPoseGait}); the dataset is available from
Science Data Bank
(\url{https://www.scidb.cn/en/detail?dataSetId=8ec62efd66a544939e821edeccc1f35c}).
Gait3D and its 2D poses are available from the dataset authors
(\url{https://gait3d.github.io/}) under their licence. Neither dataset is
redistributed. Identity splits, preprocessing, corruption seeds, per-probe
outcomes and the scripts that regenerate every table and figure are available
with the code at \url{https://github.com/likwifi/oap-supcon}.

\section*{Acknowledgment}
The authors used Claude (Anthropic) to assist with analysis scripting,
consolidation of experimental results and editing of the manuscript text. The
authors reviewed and edited all such content and take full responsibility for
the article.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""


TBIOM_EDITS = [
    # The paired-difference table moves to the supplement (Table S13) to fit 10 pages.
    ("""Table~\\ref{tab:casia5} reports the five-seed CASIA-B cohort and
Table~\\ref{tab:paired} the paired differences from the metric baseline.""",
     """Table~\\ref{tab:casia5} reports the five-seed CASIA-B cohort, and
Supplementary Table~S13 gives every paired difference from the metric baseline."""),
    # The two-part matching rule does not fit one IEEE column.
    ("""\\begin{equation}
 s_p(i,j)=\\frac{\\sum_p r_{i,p}r_{j,p}\\,v_{i,p}^{\\top}v_{j,p}}{\\max\\left(\\sum_p r_{i,p}r_{j,p},10^{-8}\\right)},\\qquad
 s(i,j)=(1-\\lambda)\\,s_g(i,j)+\\lambda\\, s_p(i,j),
 \\label{eq:matching}
\\end{equation}""",
     """\\begin{align}
 s_p(i,j)&=\\frac{\\sum_p r_{i,p}r_{j,p}\\,v_{i,p}^{\\top}v_{j,p}}{\\max\\left(\\sum_p r_{i,p}r_{j,p},10^{-8}\\right)},\\\\
 s(i,j)&=(1-\\lambda)\\,s_g(i,j)+\\lambda\\, s_p(i,j),
 \\label{eq:matching}
\\end{align}"""),
    ("\\includegraphics[width=0.62\\textwidth]{fig1.pdf}", "\\includegraphics[width=\\columnwidth]{fig1.pdf}"),
    ("\\includegraphics[width=\\textwidth]{fig3.pdf}", "\\includegraphics[width=0.82\\textwidth]{fig3.pdf}"),
    ("\\includegraphics[width=\\textwidth]{fig4.pdf}", "\\includegraphics[width=0.82\\textwidth]{fig4.pdf}"),
]

SUPPLEMENT_EDITS = [
    ("\\large Training parts versus matching parts in pose-based gait recognition}",
     "\\large Part supervision versus part matching for pose-based gait recognition\\\\ under missing joints}"),
    ("(main Table~3), so the main text does not", "(Table~S13), so the main text does not"),
    ("lists the values plotted in main Figure~3:", "lists the values plotted in main Fig.~3:"),
    ("checkpoint-control analysis in main Table~4 re-encodes", "checkpoint-control analysis in main Table~III re-encodes"),
    ("single-view average of main Table~2 into", "single-view average of main Table~II into"),
    ("Latency in main Section~5.6 was", "Latency in main Section~V-F was"),
    ("main Figures~1, 3 and~4 and every generated table.", "main Figs.~1, 3 and~4 and every generated table."),
    ("Main Table~5 reports means.", "Main Table~IV reports means."),
    ("\\section{Reproduced baselines and provenance}",
     """\\section{Paired objective comparisons}
\\label{sec:s-paired}

Table~S13 lists every paired Rank-1 difference from the metric baseline with
the global descriptor, for the five-seed and three-seed CASIA-B cohorts and the
Gait3D development split; the main text quotes the principal entries.

\\begin{table}[!htbp]
\\centering
\\caption{Paired Rank-1 difference from the metric baseline (percentage points;
crossed 95\\% bootstrap interval over seeds and identities), global descriptor.
The three-seed column is the earlier factorial on CASIA-B; Gait3D uses its
300-identity development split. Dashes mark objectives absent from a cohort. CE
denotes cross-entropy.}
\\label{tab:s-paired}
{\\tablesetup\\footnotesize
\\input{tables/tab_paired.tex}}
\\end{table}

\\section{Reproduced baselines and provenance}"""),
]


def apply(text, edits, name):
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit(f"{name}: expected exactly one match for {old[:70]!r}, found {text.count(old)}")
        text = text.replace(old, new, 1)
    return text


def drop_float(text, input_name):
    i = text.index("\\input{tables/" + input_name + "}")
    a, b = text.rfind("\\begin{table}", 0, i), text.index("\\end{table}", i) + len("\\end{table}")
    return text[:a] + text[b:].lstrip("\n")


def main():
    for name in ("fig1.pdf", "fig2.pdf", "fig3.pdf", "fig4.pdf", "figS1.pdf", "figS2.pdf", "references.bib",
                 "supp_factorial_tables.tex"):
        shutil.copy2(PR / name, HERE / name)
    source = (PR / "main.tex").read_text()
    abstract = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", source, re.S).group(0)
    keywords = ("\\begin{IEEEkeywords}\nBiometrics, gait recognition, occlusion robustness, part-based matching, "
                "pose estimation, supervised contrastive learning.\n\\end{IEEEkeywords}\n\n")
    body = source[source.index("\\section{Introduction}"):source.index("\\section*{CRediT")]
    body = drop_float(body, "tab_paired.tex")
    body = apply(body, TBIOM_EDITS, "main")
    body = body.replace("\\begin{table}[!htbp]", "\\begin{table*}[!t]").replace("\\end{table}", "\\end{table*}")
    # Figure 1 (summary) stays one column; the others span both columns.
    figures = re.split(r"(\\begin\{figure\}\[!htbp?\].*?\\end\{figure\})", body, flags=re.S)
    for k, part in enumerate(figures):
        if part.startswith("\\begin{figure}") and "fig1.pdf" not in part:
            part = re.sub(r"\\begin\{figure\}\[!htbp?\]", r"\\begin{figure*}[!t]", part)
            figures[k] = part.replace("\\end{figure}", "\\end{figure*}")
        elif part.startswith("\\begin{figure}"):
            figures[k] = re.sub(r"\\begin\{figure\}\[!htbp?\]", r"\\begin{figure}[!t]", part)
    body = "".join(figures)
    if "tab:paired" in body:
        raise SystemExit("dangling reference to the paired table")
    (HERE / "main.tex").write_text(PREAMBLE + abstract + "\n\n" + keywords + body + BACK)
    supplement = apply((PR / "supplementary.tex").read_text(), SUPPLEMENT_EDITS, "supplement")
    (HERE / "supplementary.tex").write_text(supplement)
    print("wrote main.tex and supplementary.tex")


if __name__ == "__main__":
    main()
