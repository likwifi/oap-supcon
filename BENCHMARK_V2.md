# Benchmark V2: implementation and execution

This is a stronger **candidate pipeline**, not a claim of first place. It needs controlled GPU training and official evaluation. Existing CASIA-B scores are from the previous code and are not scores for this model.

**What changed**

- Confidence is preserved as an input/reliability channel. Geometry is masked only by binary presence in augmentation, cropping, speed perturbation and evaluation corruption.
- Long clips use contiguous windows; only short sequences repeat. Low-confidence pelvis centering now divides by the actual root weight.
- The `multistream` backbone derives joint, bone and two-lag velocity inputs. Missing endpoints cannot create fake bones or velocities. Six residual graph blocks use bounded learned edges and temporal dilations 1/2/4. Independent part heads and normalized identity classification replace the reference model's shared part head and unconstrained classifier. The CASIA-B development model has 1,687,570 parameters.
- `oap_v2` directly trains the retrieval embedding with batch-hard triplet loss, alongside smoothed CE, a clean CE anchor, a smaller global SupCon term and part contrast. Temporal loss is disabled in this candidate; the existing temporal variants remain available.
- The two main views independently mix clean and partial evidence. Each has 30% probability of retaining the generically augmented clean view. This is motivated by the earlier two-partial ablation, not yet validated on the corrected pipeline.
- Part loss excludes absent parts from positives and negatives, and normalizes reliability weights to avoid shrinking the objective as the curriculum becomes harder. This intentionally changes the old Equation-8 implementation; paper equations and labels must reflect the new objective.
- `reliable_parts` matching uses shared part visibility, a separately configured global/part weight, and global fallback when no parts overlap. It does not score an absent part's projection bias. Global-only results remain available for every method.
- Main and temporal augmentation use separate RNG streams. Temporal auxiliary passes do not update BatchNorm running statistics. Body-part-mask and pairing ablations apply to the temporal branch too.
- The training recipe uses 16 identities × 4 sequences, encourages different camera views, has cosine learning-rate decay with warmup, gradient clipping, CUDA mixed precision and periodic checkpoints. BF16 is preferred where available; FP16 uses dynamic loss scaling.
- Evaluation masks use fixed per-sequence seeds, independent of training seed, device and batch size.
- Checkpoints and histories are saved before final evaluation. Configurations record effective epochs/realizations, actual train/validation identities, source-content hash and a checksum including protocol metadata.
- Aggregation separates architecture, configuration, source, data and validation/test stage; it excludes duplicate seeds according to an explicit first/last/error policy. New results go to `results_v2/`.

Joint/bone/motion inputs and a scheduled optimizer are informed by the [official FastPoseGait GaitGraph2 recipe](https://github.com/BNU-IVC/FastPoseGait/blob/main/configs/gaitgraph2/gaitgraph2.yaml). This implementation is not a reproduction of that architecture or its published accuracy.

**Development protocol**

The `benchmark_v2` preset defaults to validation evaluation. CASIA-B automatically holds out these 12 identities from the original 74 training identities, using fixed split seed 2026:

`002, 006, 012, 025, 026, 032, 043, 046, 054, 059, 061, 072`

The remaining 62 identities train the model. Within held-out identities, NM-01..04 are validation gallery and NM-05..06/BG/CL are validation probes. The NPZ is not modified. Original test identities 075–124 are not used by development evaluation or checkpoint selection. Other real datasets must provide explicit `val_gallery` and `val_probe` roles on held-out identities.

Every five epochs, checkpoint selection uses the global descriptor and:

`0.75 × clean validation Rank-1 + 0.25 × mean(body-part 0.3, random-joint 0.3 validation Rank-1)`

For CASIA-B these values use single-view galleries, average conditions, and exclude identical-view pairs. The exact metric and masks are fixed in the recipe. Changing selection weights, model size or retrieval mixture creates another development configuration. Choose among them on validation, not on final test scores. Because earlier test results have already informed this project, independent confirmation on another dataset remains necessary for a strong generalization claim.

**Run one development experiment**

From the repository directory on the GPU machine:

```bash
export PYTHONPATH="$PWD/src"
export OAP_DATA_ROOT=/cluster/datasets/oap_pose
python -m oap_supcon.cli run \
  --dataset casia_b_pose --preset benchmark_v2 \
  --method oap_v2 --seed 11 --device cuda
```

Default budget: 160 epochs, 60-frame training clips, full-sequence evaluation, five corruption realizations. `--epochs` overrides the actual recorded budget. For a smaller development screen, set it explicitly and compare all controls at the same budget. The preset is explicit so old method names remain usable for corrected-pipeline controls.

**Controlled comparison**

The new SLURM development array runs three seeds for each of:

| Method | Purpose |
|---|---|
| `ce` | Plain identity classification on the same multistream backbone |
| `masked_ce` | Existing random-dropout classification control |
| `matched_ce` | Same mixed partial views and clean CE anchor as OAP V2 |
| `metric_baseline` | Matched CE plus triplet loss in retrieval space |
| `oap_v2` | Metric baseline plus global and part supervised contrast |

```bash
sbatch slurm/12_benchmark_v2_dev.slurm
```

These jobs request one GPU and 32 GB host RAM each, with three concurrent jobs. Use the cluster's compatible partition/nodelist as appropriate. No jobs were submitted by the local code-change task.

Use global-only results as the common backbone comparison: baseline part heads are not trained by part losses, so their part-descriptor scores are not a sufficient trained-head control. Evaluate `reliable_parts` as an additional system variant; `evaluation.part_weight` is fixed at 0.5 initially and must be selected on validation before final testing.

After development, freeze the chosen recipe and epoch budget. Then `--final --epochs <budget>` disables the development holdout, trains on all 74 original training identities, and evaluates the official test set without checkpoint selection on test:

```bash
# Replace 120 with the budget selected and frozen from validation.
python -m oap_supcon.cli run \
  --dataset casia_b_pose --preset benchmark_v2 --method oap_v2 \
  --seed 11 --device cuda --final --epochs 120
```

For the five-seed final controlled matrix, export the frozen budget as `OAP_FINAL_EPOCHS` and submit `slurm/13_benchmark_v2_final.slurm`. The script requires that value rather than inventing a test-selected budget.

**Artifacts and aggregation**

- `checkpoint-last.pt`: periodically saved training model.
- `checkpoint-best.pt`: best validation model, when validation is enabled.
- `checkpoint.pt`: selected model, written before the long evaluation.
- `history.json`: loss components, training accuracy, reliability, learning rate, gradient norms and validation measurements, updated each epoch.
- `config.yaml`, `metrics.json`, `probe_outcomes.npz`: frozen configuration, final scores and paired statistical outcomes.

The last checkpoint is a recovery/inference artifact; exact optimizer-state training resume is not implemented. Final runs without validation use the predetermined final epoch. CUDA kernels and mixed-precision execution have not been exercised on this CPU-only host.

```bash
python -m oap_supcon.cli aggregate --results-root results_v2 --duplicates error
```

If an experiment was deliberately rerun with the same seed, choose `--duplicates first` or `last`; selection is by run timestamp/name, never score. Excluded runs are listed in `duplicate_runs.csv`. Saved legacy results can be summarized into another directory without changing their original tables:

```bash
python -m oap_supcon.cli aggregate --results-root results \
  --output audits/benchmark_v2/reaggregated_legacy --duplicates first
```

Synthetic corruption severities retain the existing nominal definitions. Body-part masks remove whole anatomical groups and dynamic masks are temporally smoothed, so nominal severity is not always the exact removed-joint fraction. Synthetic corruption remains applied to normalized poses; it is not a substitute for real detector occlusion. The official single-view-gallery rows currently cover clean NM/BG/CL; synthetic robustness rows use the explicitly labeled pooled-gallery protocol.

**Local verification**

**84 tests passed** on the current source revision. `audits/benchmark_v2/verification.json` records the checked source hash and limits of verification.

`OMP_NUM_THREADS=2 python3 -m pytest -q` exercises all 20 method configurations, the new model, loss stability, confidence semantics, validation separation, retrieval and aggregation. A two-epoch end-to-end development test guards every evaluated index against test-set access.

The full 1.69M-parameter architecture also completed three optimizer steps on eight real CASIA-B **training** sequences, including synthetic occlusion and all candidate losses, with finite gradients. Its integration record is `audits/benchmark_v2/integration.json`. Those steps are software verification only, not a recognition experiment or an accuracy improvement claim.

```bash
PYTHONPATH=src python3 scripts/verify_benchmark_recipe.py --device cpu --steps 3
```

The next evidence needed is the controlled GPU development matrix. A successful implementation test cannot establish a ranking against published models.

**Factorial diagnosis after the development matrix**

The initial three-seed results indicate that `metric_baseline` is stronger than
`oap_v2`, and that `reliable_parts` harms OAP while helping the controls. Diagnose
that interaction before any final-test run:

```bash
sbatch slurm/14_benchmark_v2_factorial.slurm
```

The array runs the 2x2 global-SupCon/part-SupCon factorial plus a trained
part-head triplet control and an ungated OAP control. The
`benchmark_v2_factorial` recipe writes to `results_v2_factorial/` and records
per-part reliability, presence, embedding variance, and shared-encoder loss
gradient cosines. Aggregate it independently; do not mix these diagnostic runs
with the original development cohort.

```bash
python -m oap_supcon.cli aggregate \
  --results-root results_v2_factorial --duplicates error
```
