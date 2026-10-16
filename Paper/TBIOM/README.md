# IEEE T-BIOM manuscript

Generated from `../PatternRecognition/main.tex` and `supplementary.tex` by `derive_from_pr.py`, which applies the IEEE conversion and the T-BIOM layout changes listed in it. Edit the text in the PR folder and rerun the script. Each change must match exactly once, so the script stops if the PR text has moved on. The cover letter is maintained here by hand.

**Title:** "Part Supervision Versus Part Matching for Pose-Based Gait Recognition Under Missing Joints".

## Upload set

`submission/` (also zipped as `TBIOM_submission.zip`):

| File | Purpose |
|---|---|
| `manuscript.tex` | LaTeX source (IEEEtran journal). Tables are inlined and the bibliography embedded, so it compiles with `IEEEtran.cls` alone |
| `fig1.pdf` … `fig4.pdf` | Figures 1–4, vector PDF |
| `manuscript.pdf` | Compiled manuscript: **10 pages**, the limit for T-BIOM regular papers |
| `supplementary_material.pdf` | Tables S1–S13, Figures S1–S2, 10 pages (no page limit) |
| `cover_letter.pdf` | Cover letter |

Formal checks:
- Abstract: 230 words; IEEE allows up to 250.
- Index Terms: 6.
- References: 52, IEEEtran style.
- Build: no overfull boxes, no undefined references.

## Differences from the Pattern Recognition version

**Format:** IEEE two-column. Wide tables and figures span both columns, and intervals print on one line instead of stacked.

**Title and keywords:** a T-BIOM title that leads with missing joints, and IEEE Index Terms.

**Back matter (IEEE conventions):**
- Funding goes in the first-page footnote.
- The AI-use disclosure goes in the Acknowledgment, as IEEE policy requires.
- An Ethics Statement is added, as T-BIOM requires for human-subject data.
- A Data and Code Availability section is added.
- There are no highlights, CRediT section or Elsevier declarations.

**To fit 10 pages:**
- The paired-difference table (PR Table 3) moved to the supplement as Table S13.
- Figure 1 (summary) is one column wide; Figures 3 and 4 are drawn at 82% width.

The scientific content is identical, and tables are generated from the same `evidence_pr.json`. Running `../PatternRecognition/build_pr_evidence.py` rewrites both `../PatternRecognition/tables/` and `tables/` here.

## Rebuild

1. From the project root, regenerate the evidence and tables:

   ```bash
   .venv_plot/bin/python Paper/PatternRecognition/build_pr_evidence.py
   ```

2. Run `python Paper/TBIOM/derive_from_pr.py`. It copies figures and regenerates `main.tex` and `supplementary.tex`.
3. Compile `main.tex` (with BibTeX), `supplementary.tex` and `cover_letter.tex`.
4. Assemble the package:

   ```bash
   python Paper/TBIOM/make_submission.py <build-dir>
   ```

## Before submitting: author checks

1. **Submit to one journal at a time.** The Pattern Recognition and T-BIOM versions are the same paper. Both cover letters state that the paper is not under consideration elsewhere. Also settle the IJCV submission status first.
2. **Ethics statement.** It says only public datasets were used and no new human data were collected. Confirm with your institution whether T-BIOM's rule needs an IRB exemption reference. If one exists, add the institution, board and date to the manuscript and to the submission form.
3. **Review mode.** T-BIOM allows optional double-anonymous review. The current file names the authors and the GitHub repository. For anonymous review, remove the author block, footnotes and repository link, and refer to the code anonymously.
4. **AI disclosure** (Acknowledgment). Confirm the wording.
5. **Final version only:** IEEE Transactions require author biographies (with photos), ORCID iDs and the IEEE copyright form. Pages beyond 10 after final layout incur overlength charges.
6. **Conflicts of interest.** Declare any with the Editor-in-Chief, Associate Editors-in-Chief or Associate Editor in the submission system.
7. **Code repository.** Make it public, include the regeneration scripts and `scripts/fpg_part_matching.py`, and port the two cluster-side fixes noted in the Pattern Recognition README.
