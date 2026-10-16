# Revision runbook — 8 October 2026

This runbook covers the code half of the revision plan: the bootstrap fix, mAP
intervals, the Gait3D comparison and final jobs, the five-seed CASIA-B study,
the FastPoseGait baselines and the two analyses. No GPU job has been submitted.

## Decide before submitting: freeze the architecture

Every job below defaults to the **V2 multistream** backbone, which produced all
existing Gait3D and factorial evidence. The revision plan claims that the V3
adaptive graph scores 86–88 against 81–83 on CASIA-B validation. That is not
what `results_v3_dev` shows. The adaptive graph was the weakest V3 arm (80.6
global clean against 83.0 for the V2-equivalent control). The 86–88 range
belongs to mean-plus-max pooling (86.4, three seeds) and part refinement
(88.0, one seed). Every architecture interval includes zero. Keeping V2 needs no
further runs. Switching to pooling needs a recipe change before any job below
runs, because every job would otherwise restart afterwards.

Source revision: `11b575d0…6982a97` (`source_fingerprint`). Any later edit under
`src/oap_supcon/` changes it, and analyzers refuse to pool runs from different
revisions. Sync this tree to the cluster and run the tests before submitting:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python3 -m pytest -q
```

## What changed

| Item | Change |
|---|---|
| Bootstrap | `scripts/bootstrap_stats.py` resamples seeds and identities as crossed factors (one identity draw shared by all seeds). The factorial, descriptor-control, head-control and Gait3D analyzers previously drew identities separately per seed, which understated intervals. `Paper/generated` and `results_descriptor_controls/analysis.json` were regenerated. |
| mAP | `_evaluate` now stores per-probe average precision (`ap__` keys in `probe_outcomes.npz`), so mAP gets the same paired intervals as Rank-1. Runs created before this revision have Rank-1 intervals only. |
| Comparison | `scripts/compare_runs.py`: methods × descriptors, Rank-1 and mAP, crossed intervals, method and descriptor contrasts, for any dataset. |
| Epoch budget | `scripts/choose_epoch_budget.py` applies a predeclared rule to validation curves. |
| Gait3D jobs | Recipe `gait3d_v2_comparison`; `slurm/42` (validation comparison); `slurm/41` (final, any methods). |
| CASIA-B | Recipe `casia_validation_v2` (`allow_final: false`, enforced by the CLI); `slurm/18` (7 methods × 5 seeds); `slurm/19` (descriptor and random-head controls, 5 seeds or Gait3D). |
| Baselines | `scripts/prepare_fastposegait.py --dataset {casia_b_pose,gait3d} --stage {validation,final}`; `slurm/43` runs one model. |
| Occlusion | Height-matched primary endpoint in `scripts/analyze_gait3d_occlusion.py`. |
| Missingness | `scripts/analyze_actual_missingness.py` regenerates every evaluation mask and plots accuracy against the evidence actually removed. |

The corrected intervals widened; three conclusions now include zero:

| Comparison | Before (nested) | Now (crossed) |
|---|---|---|
| `oap_v2` vs baseline, reliable parts, clean | −5.18 [−9.34, −0.63] | −5.18 [−10.69, +0.08] |
| Frozen trained part heads vs global, clean | [0.00, 3.24] | [−0.38, 3.75] |
| Reliable vs uniform weighting, robust | [0.02, 0.22] | [−0.02, 0.24] |

The IJCV manuscript numbers already came from the crossed bootstrap in
`Paper/IJCV/build_evidence.py` and are unchanged.

## Job order

### 1. CASIA-B controlled study (validation only, cheap)

```bash
sbatch slurm/18_casia_validation_5seed.slurm      # 35 tasks, ~1–2 GPU-h each
```

After completion:

```bash
python3 scripts/compare_runs.py results_casia_validation_v2 --dataset casia_b_pose \
  --methods ce,masked_ce,matched_ce,metric_baseline,metric_global_supcon,metric_part_supcon,oap_v2 \
  --seeds 11,22,33,44,55
python3 scripts/analyze_actual_missingness.py results_casia_validation_v2 --dataset casia_b_pose \
  --methods ce,matched_ce,metric_baseline,oap_v2 --seeds 11,22,33,44,55
sbatch slurm/19_descriptor_controls_5seed.slurm   # random-head control, 5 seeds
```

### 2. Gait3D validation comparison

```bash
sbatch slurm/42_gait3d_comparison_gpu.slurm       # ce, matched_ce, metric_baseline, oap_v2 × 3 seeds
```

`metric_baseline` is rerun here so all four methods share one source revision.

```bash
python3 scripts/compare_runs.py results_gait3d_v2_comparison --dataset gait3d \
  --methods ce,matched_ce,metric_baseline,oap_v2 --seeds 11,22,33
```

### 3. Fix the Gait3D epoch budget

The rule, applied to the existing metric_baseline logs (array 700), returns
**160 epochs** (`audits/gait3d/epoch_budget_from_logs.json`). The smoothed
across-seed curve is flat from epoch 105 to 160: the last epoch is 0.17 points
below the smoothed peak, and the late slope is +0.1 points per 10 epochs.
Seed 33's best at epoch 160 is a single high point (0.350 against neighbours of
0.331–0.340), not a rising curve. Confirm it on every method of the
comparison cohort, then use one common budget:

```bash
for method in ce metric_baseline; do
  python3 scripts/choose_epoch_budget.py results_gait3d_v2_comparison/gait3d_multistream_${method}_seed*
done
```

### 4. Gait3D final confirmation (once)

```bash
export OAP_FINAL_EPOCHS=160                       # only if step 3 confirms it
export OAP_FINAL_METHODS=metric_baseline,ce       # chosen model + CE baseline
sbatch slurm/41_gait3d_final_gpu.slurm            # 10 tasks; other method counts need --array
python3 scripts/compare_runs.py results_gait3d_v2_comparison_final --dataset gait3d \
  --methods ce,metric_baseline --seeds 11,22,33,44,55
python3 scripts/analyze_gait3d_occlusion.py results_gait3d_v2_comparison_final \
  --seeds 11,22,33,44,55 --output results_gait3d_v2_comparison_final/occlusion_strata
```

The occlusion analyzer reads `metric_baseline` runs. Its primary endpoint is
height-matched (see `data/gait3d/README.md`).

### 5. Official baselines on the same data

Reproduce the published CASIA-B scores first (GPGait reports 93.6 / 80.2 / 69.3):

```bash
python3 scripts/prepare_fastposegait.py --dataset casia_b_pose --stage final \
  --models gpgait,gaitgraph2 --output external_runs/fpg_casia_final
python3 scripts/prepare_fastposegait.py --dataset gait3d --stage final \
  --output external_runs/fpg_gait3d_final
```

`slurm/43_fastposegait_gpu.slurm` trains and then tests one model from its
preparation directory. The framework reads `configs/default.yaml` and, for
Gait3D, the probe list `datasets/Gait3D/Gait3D.json` relative to its working
directory; the exporter writes both there. Submit once per model:

```bash
sbatch --export=ALL,OAP_FPG_DIR=external_runs/fpg_gait3d_final,OAP_FPG_CONFIG=gpgait.yaml \
  slurm/43_fastposegait_gpu.slurm
```

Set `OAP_FPG_PYTHON` if FastPoseGait's dependencies live in a different
environment from the project's.

Each template evaluates its fixed final iteration; `with_test` is disabled so
no checkpoint is chosen on test. The framework seeds every run identically, so
each baseline is one official run.

Gait3D export differences from the official pretreatment are listed in
`manifest.json`: numeric rather than lexicographic frame order, and
confidence clipped to 1 (47,614 values). Only GPGait and GaitGraph2 have
official Gait3D recipes. GPGait++ and SkeletonGait have none here, so a Gait3D
state-of-the-art claim is not supported by these jobs.

### 6. Efficiency table

```bash
sbatch --export=ALL,CHECKPOINT=$PWD/<run>/checkpoint.pt slurm/70_efficiency_gpu.slurm
```

## Not done here

- Rewriting the paper's claim and structure (writing, not code).
- Varying the number of body parts, and part matching on GPGait features.
- Annotating the Gait3D manual-audit sheet (`manual_audit_sheet.csv`).
