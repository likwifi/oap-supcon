# Pattern Recognition manuscript

Prepared on 14 October 2026 from `../IJCV/main.tex` (2 October revision). This
version adds the cluster runs of 8–14 October and is formatted for Elsevier's
*Pattern Recognition* (elsarticle, single column, double spaced, numbered
references).

## Upload set

`submission/` (also zipped as `PR_submission.zip`):

| File | Editorial Manager item |
|---|---|
| `manuscript.tex` | LaTeX source. Tables are inlined and the bibliography is embedded, so it compiles with `elsarticle.cls` alone |
| `fig1.pdf` … `fig4.pdf` | Figures 1–4, vector PDF |
| `manuscript.pdf` | Compiled manuscript, 34 pages (limit 35) |
| `highlights.txt` | Highlights: 5 bullets, 72–77 characters each (limit 85) |
| `supplementary_material.pdf` | Supplementary material: Tables S1–S12, Figures S1–S2, 9 pages |
| `cover_letter.pdf` | Cover letter |

Formal checks:
- Abstract: 230 words (limit 250).
- Keywords: 7 (limit 7).
- References: all 52 cited.
- Build: no overfull boxes, no undefined references.

## What changed from the IJCV version

**Results:**
- The five-seed CASIA-B cohort is now the primary objective comparison: 7 objectives, including three CE controls.
- The three-seed factorial moved to the supplement. Its part-SupCon penalty (−6.12 [−10.29, −2.00]) did not replicate (−1.27 [−5.24, 2.74]), and the text says so.
- Raw part matching is now reported for every trained objective (Figure 3, Table S6). This closes an IJCV limitation.
- Added the Gait3D development and test evaluation. The test identities were evaluated once, with the recipe frozen beforehand: raw parts give +1.99 [0.61, 3.38] Rank-1 (Table 6).
- Added GPGait and GaitGraph2 reproduced on identical data with official FastPoseGait configurations (Tables 5–6).
- Added the Gait3D detector-confidence stratification (a null result).
- Added the fusion-weight transfer result: the best CASIA-B weight erases the Gait3D gain (Table S5).
- Added measured latency, plus nominal-versus-removed corruption fractions (Table S4).
- Added a transfer test on GPGait (Section 4.5, Table 7, Table S9). GPGait's trained part branches add 9.1 (CASIA-B) and 5.2 (Gait3D) points, while raw part matching lowers its accuracy. The abstract, contributions, Discussion and Conclusion now say the source of a part-aware gain is encoder-dependent. Features come from `scripts/fpg_part_matching.py`; results are in `results_fpg_part_matching/`.
- Moved the corruption-family diagram to the supplement (Figure S2) to stay within 35 pages.
- Adopted presentation elements from GPGait (ICCV 2023):
  - a results figure up front: Figure 1, the test-set Rank-1 change for our encoder against GPGait;
  - one main test-set comparison, Table 5, which replaces three tables (their details moved to Tables S10–S12);
  - a separate Section 4, "Experimental setup";
  - silhouette-comparison and Outlook paragraphs in the Discussion;
  - an ISO/IEC 19795-1 citation in the ethics paragraph;
  - the agreement with GPGait's own partition-mask ablation, in Section 5.5.
  Two references were added: ISO/IEC 19795-1 and OU-MVLP-Pose.
- Figure files are now `fig1` (summary), `fig2` (framework, from `figure_sources/fig2.tex`), `fig3` (forest plot) and `fig4` (severity curves).

**Format:**
- Converted to Elsevier `elsarticle`.
- Added highlights, the CRediT statement, and the Elsevier competing-interest, data-availability and generative-AI declarations.

## Rebuild

Run from the project root. NumPy, PyYAML and Matplotlib are needed; on the cluster use `.venv_plot/bin/python`:

```bash
.venv_plot/bin/python Paper/PatternRecognition/build_pr_evidence.py
```

This reads the result directories (read only) and checks seeds, probe counts and hashes. It then rewrites:
- `evidence_pr.json`;
- `fig1.pdf`, `fig3.pdf` and `fig4.pdf`;
- `tables/*.tex`.

Every number in the tables comes from `evidence_pr.json` through `make_tables.py`. Numbers in the running text were checked against the same file.

Compile `main.tex`, `supplementary.tex` and `cover_letter.tex` with `latexmk -pdf` (BibTeX is needed for `main.tex`). Then run:

```bash
python Paper/PatternRecognition/make_submission.py <build-dir>
```

The build directory must contain `main.pdf`, `main.bbl`, `supplementary.pdf` and `cover_letter.pdf`. The script assembles `submission/` and the zip.

To rebuild the TikZ figures, compile the sources in `figure_sources/` (`fig2`, `figS1`, `figS2`) with `pdflatex`. The PDFs are written next to `main.tex`.

## Other versions

An IEEE T-BIOM version of the same paper is in `../TBIOM/`. It is generated from this folder's `main.tex` and `supplementary.tex` by `../TBIOM/derive_from_pr.py`, so edit the text here and regenerate. Submit to one journal at a time.

## Before submitting: author checks

1. **Prior submission.** The cover letter states the paper is not under consideration elsewhere. An IJCV version exists (`../IJCV/IJCV_submission.zip`). If it was submitted and is still under review, withdraw it first, or tell the editor.
2. **CRediT roles** (end of `main.tex`) are a draft based on the corresponding author writing the code and analysis. Confirm or edit them.
3. **Generative-AI declaration.** Confirm the tool and the purposes listed.
4. **Code repository.** The data-availability statement points to `https://github.com/likwifi/oap-supcon`. Make sure it is public and includes the regeneration scripts. It must also include two fixes made on the cluster on 12 October that the Mac copy lacks: `slurm/70_efficiency_gpu.slurm` and `scripts/analyze_descriptor_controls.py`.
5. **Gait3D licence and URL.** Check that the dataset licence permits this use and that `https://gait3d.github.io/` is the address you want cited.
6. **Editorial Manager items** not prepared here: suggested reviewers, ORCID iDs, and author biographies or photos if the system asks for them.
