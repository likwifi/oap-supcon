# Code fixes

Two rounds of defects found in the reference implementation, the evidence for
each, the changes made, and their measured effect.

- [Round 1 -- padding-as-occlusion and temporal-loss false negatives](#round-1--padding-as-occlusion-and-temporal-loss-false-negatives)
- [Round 2 -- discarded part descriptors, confidence in the geometry, padding in the trunk](#round-2--discarded-part-descriptors-confidence-in-the-geometry-padding-in-the-trunk)

---

## Round 1 -- padding-as-occlusion and temporal-loss false negatives

Record of two defects found in the reference implementation, the evidence for
each, the changes made, and their measured effect on the CASIA-B core matrix.

| | |
|---|---|
| Pre-fix code | `74007d3` |
| Fix | `c1dc01c` |
| Results matrix re-run at | `15d410c` |
| Pre-fix results archived in | `results_prefix_74007d3/` |
| Date | 2026-09-17 |

All 47 pre-fix runs recorded `git_commit: "not-a-git-checkout"`; the repository
did not exist when they were executed. The 25 post-fix runs all carry
`15d410c`, so the two generations are distinguishable from their artifacts
alone.

---

### Defect 1 — padding was treated as occluded evidence

#### What was wrong

`convert_casia_b_hrnet` pads every sequence to `max(record.frames)` and records
the true length in `frame_counts`. For CASIA-B that is **T = 306 with a mean
real length of 96.7 frames, so 68% of every tensor is padding.**

`frame_counts` was read only by the audit reporter. Training, augmentation,
corruption and evaluation all ignored it and treated the zero-padded tail as
genuinely occluded body evidence.

#### Consequences, measured on the real CASIA-B tensors

**1. The Equation 11 reliability weight measured clip length, not occlusion.**
`ω` averaged over the padded buffer, so a *fully visible* part scored
`ω ≈ 0.27` and correlated **0.985** with sequence length. Pairwise gating
`ω_u · ω_v ≈ 0.078` then scaled the part loss to ~8% of nominal. The logged
training history shows the result:

```
epoch 120: ce_loss 2.91   global_loss 3.41   part_loss 0.15   temporal_loss 1.59
```

At `part_weight: 0.5` the part term — the paper's headline contribution — was
carrying **~1% of the total objective.**

**2. Temporal corruption severity was calibrated against the padding.** The
interruption masked `severity × 306` frames, so the requested severity was not
the severity applied:

| nominal severity | real frames actually removed |
|---|---|
| 0.1 | 9.5% |
| 0.3 | 23.4% ± 30.6% (range 0–99%) |
| 0.5 | 33.7% ± 34.0% |
| 0.7 | 51.6% ± 29.2% |

The within-condition spread exceeded the between-condition difference, making
the whole temporal robustness column close to uninterpretable.

**3. Temporal-consistency positives were frequently empty.**
`temporal_crop_view` cropped the padded buffer, so **14.8% of crops were 100%
padding** and 22.4% retained under a quarter of the real joint-frames. The
InfoNCE term was regularly asked to match two all-zero sequences to distinct
identities.

#### The change

Thread the real length through `data → augment → model → experiment` and
normalise every temporal operation by it.

- `data.py` — `PoseDataset.__getitem__` returns `length` as a third element.
  **This changes the tuple arity**; `scripts/measure_efficiency.py` was updated
  to match.
- `augment.py` — added `sequence_lengths` and `frame_mask_from_lengths`;
  `corrupt`, `temporal_crop_view`, `speed_perturb` and `training_views` take an
  optional `lengths`; `_resample_crop` gained a `target` argument so a crop is
  resampled onto the sequence's own length and re-padded rather than stretched
  across the buffer; `_per_sample_occlusion` renamed to `occlusion_view` (old
  name kept as an alias).
- `model.py` — `forward` takes an optional `frame_mask`; `ω` is normalised over
  real frames so a fully visible part scores 1.0 at any clip length.
- `experiment.py` — builds `frame_mask` per batch and passes lengths into both
  training views and evaluation-time corruption.

Sequences shorter than two frames are guarded (`min_frames` is 1 in the CASIA-B
gallery). Every new argument defaults to `None` and falls back to the padded
length, so callers without padding — the smoke set, the unit tests — are
unaffected.

#### Verification

| measurement | before | after |
|---|---|---|
| ω, clean clips | 0.269 | **0.840** |
| ω, body-part severity 0.5 | 0.071 | **0.219** |
| corr(ω, clip length) | +0.985 | +0.347 |
| temporal severity 0.3 → real frames removed | 23.4% ± 30.6% | **30.0% ± 0.3%** |
| crops that are 100% padding | 14.8% | **0%** |
| part-loss share of the objective | **1.2%** | **10.2%** |

The residual +0.347 correlation between ω and clip length is genuine — longer
clips have slightly different HRNet confidence profiles — not an artifact.

---

### Defect 2 — the temporal loss fought the supervised contrastive loss

#### What was wrong

Batches are sampled `P` identities × `K` sequences by `IdentityBatchSampler`
(8 × 4), but `temporal_contrastive` used `targets = arange(N)`, treating only
the diagonal as positive. Every anchor therefore had **K−1 = 3 sequences of its
own identity among its 31 negatives**, and the term pushed them apart — exactly
what `supervised_contrastive` pulls together.

`oap_supcon` is the only core method with `temporal_weight > 0`, which is why
this degraded the full method specifically.

The crops were also drawn from clean `x`, so the term never saw occluded
evidence.

#### The change

`temporal_contrastive` takes an optional `labels` and removes same-identity
off-diagonal pairs from the denominator with `masked_fill(..., -inf)`: they
count as neither positive nor negative.

**Note on the approach.** The obvious alternative — converting those pairs into
positives by reusing `supervised_contrastive` — was rejected. It would make
`L_temp` a near-duplicate of `L_supcon` on crops, which destroys what the
`no_temporal` ablation is supposed to isolate. Masking preserves Equation 14's
crop-to-crop instance-level pairing while removing the contradiction.

The second crop now also passes through `occlusion_view`, so the temporal term
sees partial evidence rather than clean clips only.

---

### Effect on the results

The ordering that made the method look broken reversed completely.
`oap_supcon` minus `global_occlusion_supcon` (the ablation that removes the
part and temporal losses), clean-gallery rank-1, percentage points:

| corruption | severity | pre-fix | post-fix |
|---|---|---|---|
| random_joint | 0.3 | −8.57 | **+3.98** |
| random_joint | 0.5 | −9.36 | **+3.92** |
| random_joint | 0.7 | −5.24 | **+2.14** |
| dynamic_joint | 0.3 | −2.72 | **+2.72** |
| dynamic_joint | 0.5 | −5.78 | **+2.94** |
| dynamic_joint | 0.7 | −3.18 | **+1.71** |
| temporal | 0.5 | +2.34 | **+6.27** |

Seven sign flips. `oap_supcon` now wins 15 of 16 severity × family cells (the
exception, `body_part 0.7`, is −0.01). Identity-level paired bootstrap (10,000
samples, 50 identities) puts 8 of 12 seed × corruption comparisons at a 95% CI
excluding zero, with gains up to +10.74 pp. Seed variance also tightened —
`dynamic_joint 0.3` went from ±1.91 to ±0.32.

**Control:** all five `ce` seeds reproduced their pre-fix clean rank-1
bit-identically (41.94 / 42.64 / 44.33 / …, delta 0.00). `ce` uses
`augmentation: none` and `part_weight: 0`, so none of the changes touch its
clean-gallery path. This confirms the fixes are confined to the padding and
temporal paths and did not perturb the shared encoder, sampler, normalization
or retrieval code.

**Side effect worth reporting:** `generic_supcon` changed character. Clean
rank-1 rose 33.17 → 42.43 (it now gets non-degenerate temporal crops), but its
robustness collapsed to chance — 2.00 ± 0.00 at `random_joint` 0.3 and above,
exactly 1/50 identities. Best clean accuracy of any contrastive method, zero
occlusion robustness.

---

### Regression tests

The suite went 11 → 16. The original tests all built fully-visible tensors,
which is why the padding defect survived them.

| test | file | guards |
|---|---|---|
| `test_temporal_severity_is_measured_against_the_real_clip_not_the_padding` | `test_augment.py` | severity scales the real clip; padding untouched |
| `test_temporal_crop_never_samples_from_the_padding` | `test_augment.py` | no empty crops, no leak past the boundary |
| `test_speed_perturb_keeps_the_padding_boundary` | `test_augment.py` | resampling stays within `[0, L)` |
| `test_temporal_infonce_drops_same_identity_false_negatives` | `test_losses.py` | masked loss < unmasked, stays finite |
| `test_part_reliability_is_normalised_over_real_frames_not_padding` | `test_experiment.py` | ω = 1.0 for a fully visible part on a 25-of-100-frame clip |

---

### Infrastructure fixes made alongside

- **`.gitignore`** (`3010a40`) — originally listed only `.venv/`, but this
  project's virtualenv is `env/` (5.4 GB). Added `env/`, `venv/`,
  `*.egg-info/`, `results_prefix_*/`, and `$HOME/` — a 26 MB directory created
  by an unquoted `$HOME` in a venv bootstrap command.
- **`slurm/submit_main.sh`** (`08e9076`) — added `OAP_SLURM_EXCLUDE` and
  `OAP_SLURM_NODELIST`. The cluster mixes H100 (sm_90) and Volta (sm_70) nodes,
  but the installed torch 2.14.0+cu130 compiles only
  `sm_75/80/86/90/100/120`. Tasks landing on a Volta node died within ~8 s with
  `CUDA error: no kernel image is available for execution on the device`,
  taking out 22 of 25 core tasks and all 45 ablation tasks in job arrays 347 and
  352. This is the same error that killed `ce_seed44` in the September matrix,
  which passed on resubmit only because it happened to land on a Hopper node.
  Use `export OAP_SLURM_EXCLUDE=volta1,volta2`.
- **`scripts/status.sh`** (`15d410c`) — run-matrix progress. Uses `squeue -r`
  because a pending array range otherwise collapses to a single line and
  22 pending tasks read as 1.

---

### Known issues not addressed

1. **`git_commit` is stamped at run end, not run start.** `run_experiment`
   calls `git_commit()` when building the metrics dict, after training and
   evaluation. Editing the repo during a long array mis-stamps the runs — all
   25 post-fix runs recorded `15d410c` because `scripts/status.sh` was committed
   while the array was in flight, even though they started at `08e9076`. The
   call belongs next to the frozen `config.yaml` write.
2. **The encoder cannot represent anatomy.** `PoseEncoder` is
   permutation-equivariant over joints — shuffling all 17 leaves the embedding
   bit-identical (max diff 2.2e-08) — with no adjacency, no joint embedding, and
   a single shared part-projection head. The paper claims an ST-GCN / GaitGraph
   / skeleton-transformer backbone. The part-level loss now carries real
   gradient weight but still has no anatomical structure to exploit, which is
   the likeliest reason `body_part` remains the weakest effect (CI excludes zero
   in only 1 of 3 seeds).
3. **`no_visibility` and `unweighted_parts` are byte-identical configs** in
   `configs/methods.yaml`, so the ablation grid has a duplicate row. Left
   unchanged because the intended distinction is unclear — `no_visibility` may
   have been meant to ablate the encoder's visibility input channel.
4. **The paper states empirical validation is deferred to a follow-up study.**
   Two full CASIA-B matrices now exist. The text is stale.

---

## Round 2 -- discarded part descriptors, confidence in the geometry, padding in the trunk

| | |
|---|---|
| Pre-fix code | `4cbb11b` |
| Fix | `e1fa89e`, `e49bdfc` |
| Pre-fix results archived in | `results_prefix_15d410c/` |
| Date | 2026-09-18 |

Round 1 fixed how padding and the temporal loss were *used*. Round 2 covers what
was still wrong with the representation reaching the encoder, and with what came
back out of it at matching time. Every figure below is measured on the real
CASIA-B HRNet tensors.

Where round 1 could keep `ce` bit-identical as a control, round 2 cannot: it
changes the input representation for every method. The whole core matrix is
therefore re-run, and the old artifacts are archived rather than merged.

---

### Defect 3 — the part heads never reached retrieval

#### What was wrong

`_embeddings` called `model.embed`, which returns the global sequence embedding
only. The five part projections — the paper's contribution — were trained by
Equation 8 and then thrown away before the gallery was ever scored. Every
reported number was the global embedding alone, for all five methods.

#### Consequences

Concatenating the global embedding with the five L2-normalised part projections
(rescaled so the result stays unit norm) changes clean rank-1 as follows, five
seeds on the pooled-gallery protocol and two seeds on the official one:

| method | pooled, global | pooled, +parts | official NM, global | official NM, +parts |
|---|---:|---:|---:|---:|
| `ce` | 0.4227 | 0.3971 | 34.75 | 29.25 |
| `masked_ce` | 0.2985 | 0.2853 | 22.10 | 19.92 |
| `generic_supcon` | 0.4292 | 0.4133 | 33.70 | 29.09 |
| `global_occlusion_supcon` | 0.3750 | 0.3633 | 32.25 | 27.15 |
| **`oap_supcon`** | 0.3847 | **0.4582** | 31.53 | **39.20** |

The four baselines all *lose* from the change — their part projector receives no
gradient, so those blocks are a random projection that dilutes the descriptor.
Only the method that trains the part heads gains, by +7.35 pp pooled and
+7.67 pp on official NM. Parts alone, with no global block, already score 0.4497
against `ce`'s 0.4197.

This matters for the paper's first claim gate. On the global descriptor
`oap_supcon` loses to `ce` on clean rank-1 by 4.8 pp, which fails the gate. On
the part descriptor it wins, and under the official protocol it becomes the best
method overall (mean 26.69 vs `ce` 25.90).

#### The change

`_embeddings` runs one forward pass and returns both descriptors; `_evaluate`
scores and reports `global` and `global_parts` **for every method**, so the
comparison is not the proposed method being given a retrieval trick the
baselines are denied. `evaluation.descriptors` selects which to compute.

---

### Defect 4 — detector confidence was multiplied into the coordinates

#### What was wrong

`normalize_pose` computed `centered = (x - center) * visibility`. For a dataset
whose visibility channel is a binary mask that is a no-op on the visible joints.
CASIA-B's channel is HRNet heatmap confidence: **mean 0.852, SD 0.116, p1 0.409,
and never exactly zero.** So it was not masking anything; it was shrinking every
joint toward the pelvis in proportion to how confident the detector happened to
be on that joint in that frame.

#### Consequences

| measurement | value |
|---|---|
| per-joint radial shrink | **14.8% ± 11.6%** |
| p95 of that shrink | **39.1%** |
| between-sequence spread of shank/body-scale ratio | 29.3% |

The injected geometric noise is roughly half the magnitude of the anatomical
variation the encoder is meant to discriminate on, and it is correlated with
pose-estimator behaviour rather than with identity. The reference pipelines do
not do this: FastPoseGait's `GaitGraph_MultiInput` carries confidence strictly
as its own channel.

#### The change

`pose.weight_coordinates: false`. Confidence still reaches the encoder as an
input channel and still weights the masked-mean pooling; it no longer deforms
the skeleton. Joints with visibility exactly zero — padding, and anything
`corrupt` has masked — still collapse to the origin, so the occlusion semantics
are unchanged.

---

### Defect 5 — the convolutional trunk still consumed the padding

#### What was wrong

Round 1 removed padding from the masking, cropping and reliability paths. It did
not remove it from the trunk. Sequences are padded to T=306 against a mean real
length of 96.7, so **68.4% of every tensor is padding**, and `encode_joints`
convolves all 306 frames. `masked_mean` excludes padding at pooling time;
`BatchNorm2d` does not.

#### Consequences

Measured at the first BatchNorm inside the temporal trunk, over training
batches:

| | mean activation |
|---|---|
| real frames | +4.5484 |
| padded frames | +9.6265 |
| **stored BN running mean** | **+8.0301** |

`0.32 × 4.5484 + 0.68 × 9.6265 = 8.03`. The running statistics are, to three
significant figures, the padding-weighted mix — so every real activation was
normalised by statistics two thirds determined by zeros.

#### The change

`train.clip_length: 60` samples a contiguous window of real frames, wrapping
around for sequences shorter than the window rather than padding them back out.
This is also what the reference baselines do: GaitGraph2 and GaitTR both use
`frames_num_fixed: 60`. Measured zero-visibility fraction inside a training
batch goes **0.684 → 0.000**, and training compute drops about fivefold.

Evaluation deliberately keeps `clip_length: null` and scores whole sequences, so
no probe evidence is discarded; each batch is trimmed to its longest real
sequence, and BatchNorm is in eval mode there, using running statistics.

---

### Defect 6 — `clean_rank1.tex` did not report clean rank-1

`aggregate_results` took the first row per run with `severity == 0` and
`protocol == clean_gallery`. `_evaluate` emits the per-condition `official_*`
slices before the corruption families, and `np.unique` orders them BG, CL, NM —
so the table reported **the BG condition** under the heading "clean rank-1".

`ce` read 0.431 where the true clean rank-1 is 0.425; every row was the BG
column mislabelled. `robustness_auc.csv` used the correct definition, so the two
generated outputs disagreed with each other. The selection now requires the
corruption to be a synthetic family, and the table carries a descriptor column.

---

### Defect 7 — no published number was comparable

`_evaluate` pools all 11 gallery views into a single 2197-sequence gallery and
excludes identical-view pairs. Published CASIA-B and OU-MVLP numbers come from
the official protocol: per condition, a probe-view × gallery-view matrix whose
gallery is a **single** view, averaged with the identical-view diagonal excluded
(`single_view_gallery_evaluation` in FastPoseGait). These are different tasks
and the numbers are not interchangeable.

Runs on the indoor benchmarks now emit `protocol: single_view_gallery` rows for
every descriptor, from the same clean embeddings, alongside the pooled rows.
Verified against the archived `ce` seed-11 checkpoint: NM 0.3456 / BG 0.2561 /
CL 0.1655 on the global descriptor, reproducing the offline calculation.

For reference, on the same HRNet pose release (`external/FastPoseGait`
model zoo, NM/BG/CL):

| model | NM | BG | CL |
|---|---:|---:|---:|
| GaitGraph1 | 86.37 | 76.50 | 65.24 |
| GaitGraph2 | 80.29 | 71.40 | 63.80 |
| GPGait | 93.60 | 80.15 | 69.29 |

`ce` at NM 34.56 is roughly 45 points below the weakest of these. The pooled
protocol was flattering the numbers; the gap is a backbone and recipe gap, not
an evaluation artifact, and closing it is the subject of the round-3 work listed
below.

---

### Regression tests

The suite went 23 → 31.

| test | file | guards |
|---|---|---|
| `test_clip_sampling_returns_dense_windows_with_no_padding` | `test_data.py` | a sampled clip contains no padding frame |
| `test_clip_sampling_wraps_sequences_shorter_than_the_window` | `test_data.py` | short sequences repeat rather than pad |
| `test_confidence_can_be_kept_out_of_the_pose_geometry` | `test_data.py` | direction preserved, relative limb length undistorted |
| `test_absent_joints_stay_at_the_origin_without_coordinate_weighting` | `test_data.py` | zero-visibility joints still collapse |
| `test_part_descriptor_is_unit_norm_and_distinct_from_the_global_one` | `test_experiment.py` | shape, norm, and the global block's scaling |
| `test_clean_rank1_table_uses_the_overall_row_not_a_condition_slice` | `test_experiment.py` | the BG row can no longer be picked up |
| `test_single_view_gallery_excludes_the_identical_view_diagonal` | `test_experiment.py` | official protocol drops identical-view cells |
| `test_single_view_gallery_needs_more_than_one_view` | `test_experiment.py` | refuses a degenerate single-view dataset |

---

### Round-1 known issues now closed

- **`git_commit` stamped at run end** — now stamped next to the frozen
  `config.yaml`, and recorded in it, so editing the repository mid-array no
  longer mis-stamps runs.
- **`no_visibility` and `unweighted_parts` byte-identical** — `unweighted_parts`
  removed; the ablation array is `0-39`, not `0-44`.
- **`measure_efficiency.py` hardcoded `PoseEncoder`** — now builds the
  checkpoint's own backbone, times the representation the run was evaluated on
  rather than the padded buffer, and counts the part projector as inference
  weight when the run matches on part descriptors.

### Known issues still not addressed

1. **The encoder still cannot represent anatomy on the default backbone.**
   `stgcn` and `transformer` exist and have Slurm workflows, but no CASIA-B
   results. The part-level claim is currently tested on a permutation-equivariant
   trunk.
2. **No validation split.** CASIA-B here is 74 train / 50 test with no `val`
   identities, `_train` runs a fixed epoch count, and `run_sensitivity.py`
   sweeps 18 hyperparameter settings scored on the probe/test identities. That
   is tuning on test and should be moved to identities held out of the 74.
3. **The recipe is far from the reference baselines.** Batch 32 against
   GaitGraph2's 768, SupCon temperature 0.1 against 0.01, a constant learning
   rate against OneCycleLR, and 3 input channels against GaitGraph2's 15
   (joint + centre offset, velocity at two lags, bone vector + bone angles).
4. **No CASIA-B ablations have ever been run.** Only smoke-data ablations exist.
5. **SUSTech1K, the declared primary real-occlusion endpoint, has no data.**
