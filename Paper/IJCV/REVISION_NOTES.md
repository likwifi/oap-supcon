# IJCV manuscript revision — 2 October 2026

The revision improves the paper's framing and evidential accuracy using existing
experiment artifacts. No models were trained and no new benchmark observations
were introduced.

## Changes made

- Shortened the title and focused the abstract/introduction on the distinction
  between local training and local retrieval scoring.
- Replaced the broad method claim with three explicit, supported contributions.
- Condensed Related Work while retaining the existing 49 cited sources; added
  the primary FastPoseGait report to contextualize the absolute accuracy gap.
- Disclosed that earlier project runs evaluated all 50 official test identities.
  The current five-seed experiment is described as fixed-recipe within-dataset
  evaluation rather than independent confirmation.
- Recomputed development uncertainty with a crossed bootstrap: model seeds and
  identities are independently resampled, and each identity draw is shared across
  sampled model seeds. This reflects the repeated identity population across runs.
- Updated all 40 cells in the paired-comparison table and all corresponding
  descriptor-control intervals in the text. Recognition accuracy measurements
  and the official-test intervals are unchanged.
- Added Figure 4 with severity curves for four corruption families, including
  seed standard deviations and explicit nominal-severity labeling.
- Clarified overlapping anatomical regions, omitted face joints, the ungated
  presence threshold, and the distinction between projected and raw descriptors.
- Confirmed near-collapse from training logs at the selected part-triplet epochs,
  rather than relying only on the last 20 training epochs.
- Removed unsupported causal language and obsolete experimental planning prose.
- Made limitations concrete: fixed hyperparameters, global-only selection,
  missing raw-matcher comparisons across all objectives, absent random-region
  controls, normalization before masking, and prior test exposure.
- Quantified the descriptor's sixfold feature-storage increase without claiming
  measured inference latency or throughput.
- Corrected Figure 3's label from five training seeds to five mask realizations.

## Principal reanalysis results

| Development comparison | Difference (percentage points) | Crossed 95% interval |
|---|---:|---:|
| Part SupCon/global minus metric baseline/global; corruption aggregate | −6.12 | [−10.29, −2.00] |
| Raw reliability-weighted matcher minus global; corruption aggregate | +4.32 | [+2.89, +5.71] |
| Raw reliability-weighted minus uniform; corruption aggregate | +0.12 | [−0.02, +0.24] |

The main degradation and matching benefit survive the correction. The small
reliability-weighting difference is now inconclusive on the primary development
aggregate. The exact settings, numerical results, selected-epoch diagnostics,
and input artifact manifest are in evidence_crossed.json. build_evidence.py
regenerates that file and Figure 4.

## Verification

- The complete manuscript builds with the existing local Springer template and
  TeX installation; the final log contains no warnings or overfull boxes.
- The manuscript retains 19 pages, with 50 cited bibliography entries.
- All 40 development table estimates were checked against evidence_crossed.json.
- The rendered manuscript was inspected for layout and figure readability.
- IJCV_submission.zip contains the revised source, PDF, template, bibliography
  metadata/style, and all four figure PDFs at the archive root.

The built-in single-file editor cannot resolve the external Springer class;
the locally compiled PDF provides the verified preview.

## Evidence still needed for a stronger IJCV submission

1. Repeat the descriptor comparison on an established reproduced backbone.
2. Add an independent dataset and a real-occlusion endpoint.
3. Apply raw matching to all six objective checkpoints and compare size-matched
   random regions.
4. Test a small, validation-only hyperparameter sensitivity range.
5. Release the exact historical source snapshots, locked dependencies, and
   official-test analysis provenance.

These gaps require evidence beyond manuscript editing. Their disclosure improves
the paper's credibility but does not itself establish generality or acceptance.
