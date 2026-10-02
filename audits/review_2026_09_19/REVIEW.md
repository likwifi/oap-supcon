# OAP-SupCon code and experiment review — 19 September 2026

Historical review of the code before the subsequent implementation changes. See `BENCHMARK_V2.md` at the repository root for the fixes and new candidate pipeline. The numerical results below remain historical; the new model has not completed a benchmark matrix.

The project has a useful experimental structure, and its part descriptors show a real signal within the TCN experiments. However, the current results do not support the full claim that OAP-SupCon preserves clean recognition while improving body-part occlusion robustness over strong controls. Fix the augmentation semantics before interpreting another loss-weight sweep. The ST-GCN pilot is the strongest lead for improving recognition, but it still needs controls on the same backbone.

**Scope and reproducibility**

Reviewed all first-party model, data, augmentation, loss, training, evaluation and reporting modules; the conversion and experiment scripts; configurations; recent histories and outcome files; and the archived generations. Ran `python3 -m pytest -q`: **34 tests passed in 4.67 seconds**. Ran small diagnostic reproductions, including a one-batch synthetic training trace. No source implementation, saved experiment, or existing result table was modified, and no benchmark training was launched.

This directory contains a reproducible analysis script, a complete run manifest, separate summaries, exploratory paired bootstraps, diagnostics, and a figure. Run `python3 audits/review_2026_09_19/analyze.py` to regenerate the derived files. They use the saved scores, without rerunning test evaluation.

The current `results/` contains **70 completed CASIA-B runs**:

- 25 core TCN runs: five methods × seeds 11, 22, 33, 44, 55.
- 40 TCN ablation runs: full OAP plus seven ablations × the same five seeds.
- One ST-GCN OAP pilot, seed 11.
- Four representation pilots, seed 11.

The core and ablation jobs each ran full OAP. This report keeps these cohorts separate, using the earliest OAP run per seed for core and the later run for ablations, following timestamps and job logs rather than scores. Four repeated seeds have identical Rank-1 results; seed 22 differs, by up to 4.67 pp across evaluation rows. The saved metadata do not establish why. Repeated seeds are not additional independent seeds. `run_manifest.csv` records every inclusion explicitly.

**What the package is trying to establish**

You are learning pose-based gait identity descriptors that transfer from training identities to unseen people, especially when only part of a skeleton is observed. The training objective combines identity classification, global supervised contrast, reliability-weighted body-part contrast, and temporal crop consistency. Complete-to-partial training pairs are intended to teach recognition from incomplete evidence. Gallery/probe nearest-neighbor matching tests the learned descriptors on different identities.

The intended scientific contribution is the occlusion-aware objective and pairing, evaluated on a shared backbone. A better backbone can improve the system, but cannot alone establish that the proposed losses outperform their controls. The declared primary real-occlusion endpoint is SUSTech1K; the locally available data and results are CASIA-B with synthetic masking.

The module boundaries are sensible:

| Area | Files | Assessment |
|---|---|---|
| Standardization and split construction | `casia_b.py`, `data.py` | Explicit identity/view/condition metadata and frame counts are valuable. CASIA-B conversion is substantially more complete than the README suggests. |
| Training views and corruptions | `augment.py` | Central to the method; currently conflates confidence with binary presence. |
| Representation | `model.py`, `graph.py` | Common heads and a backbone factory make controlled comparisons feasible. |
| Objectives | `losses.py` | Compact and inspectable; temporal false-negative masking is implemented. |
| Training and evaluation | `experiment.py` | Reproducible artifacts and two retrieval descriptors are useful; validation and checkpointing need separation from final testing. |
| Reporting and execution | `reporting.py`, `scripts/`, `slurm/` | Useful infrastructure, but grouping does not preserve experiment identity. |

The default TCN has two temporal kernels of length three: a five-frame local receptive field at evaluation, followed by pooling. It has no joint graph or joint identity embedding. Its global descriptor is invariant to a consistent joint permutation. Fixed part pooling does retain coarse anatomical grouping, so it would be too strong to say the entire part descriptor has no anatomical information. Nonetheless, the trunk cannot directly model relationships between neighboring joints. The class called `SkeletonTransformerEncoder` uses spatial attention and temporal convolutions, despite its docstring mentioning temporal attention.

**What the recent results actually show**

All values below are Rank-1 percentages. Five-seed entries are mean ± sample standard deviation. NM/BG/CL use the official single-view-gallery protocol, excluding identical-view pairs. The subsequent clean and corrupted results use pooled gallery views, also excluding identical-view pairs. These two protocols must not be compared as if they measured the same task.

| Core method / descriptor | NM | BG | CL |
|---|---:|---:|---:|
| CE / global | 42.67 ± 1.86 | 31.19 ± 1.46 | 18.75 ± 0.31 |
| Masked CE / global | 32.13 ± 2.43 | 22.22 ± 1.61 | 11.39 ± 1.25 |
| Generic SupCon / global | 29.64 ± 0.92 | 20.25 ± 0.30 | 10.98 ± 0.33 |
| Global occlusion SupCon / global | 26.53 ± 0.99 | 18.47 ± 0.77 | 10.63 ± 0.41 |
| Full OAP / global | 26.31 ± 1.75 | 18.39 ± 1.46 | 10.50 ± 0.72 |
| Full OAP / global + parts | 30.54 ± 2.82 | 19.97 ± 1.83 | 10.52 ± 0.59 |

| Method / descriptor | Clean pooled | Body-part 0.5 | Random-joint 0.5 | Temporal 0.5 |
|---|---:|---:|---:|---:|
| CE / global | 50.65 ± 0.51 | 6.45 ± 0.33 | 2.10 ± 0.13 | 8.17 ± 1.56 |
| Masked CE / global | 40.80 ± 1.46 | 8.85 ± 0.89 | 17.78 ± 2.61 | 19.58 ± 1.79 |
| Full OAP / global | 36.10 ± 1.84 | 5.76 ± 0.28 | 13.12 ± 6.39 | 16.41 ± 2.11 |
| Full OAP / global + parts | 38.50 ± 2.78 | 7.79 ± 1.33 | 19.90 ± 5.35 | 26.67 ± 2.11 |
| Two partial views / global + parts | 38.38 ± 1.93 | 9.68 ± 0.52 | 29.24 ± 3.88 | 30.72 ± 0.89 |
| ST-GCN OAP / global — **one seed** | 59.67 | 19.72 | 30.22 | 44.70 |

The descriptor is explicit because `global_parts` has 768 dimensions versus 128 for `global`. The fully matched descriptor tables, including both gallery protocols and all severities, are in `rank1_summary.csv`. The plot uses selected descriptors to illustrate system behavior, not to isolate an individual loss effect.

The clean-performance claim fails in the current core matrix: full OAP with parts is **12.15 pp below CE global** on clean pooled evaluation, and 12.13 pp below on official NM. Parts help OAP, but do not close that gap. Body-part severity 0.5 is also lower than masked CE global, 7.79% versus 8.85%.

Using the same `global_parts` descriptor for both methods, masked CE scores 42.70% clean and 9.49% on body-part 0.5. OAP is 4.20 pp worse clean and 1.69 pp worse under that corruption. Its smaller clean-to-corrupted drop is partly a consequence of its lower clean starting point; report absolute corrupted accuracy and clean accuracy together. Existing per-seed identity bootstraps show no positive OAP body-part advantage with a 95% interval excluding zero, and one negative interval. Under random-joint masking, OAP improves on masked CE with parts by 1.86 pp on average, but only one of five per-seed intervals excludes zero positively. Temporal masking is the clearer OAP advantage, +2.41 pp with three positive intervals. These are exploratory per-seed results, not a joint five-seed confirmatory test.

The completed ablations are informative. With `global_parts`, using the full-OAP repetition from the same ablation cohort:

| Ablation | Clean | Body-part 0.5 | Random-joint 0.5 | Dynamic-joint 0.5 | Temporal 0.5 |
|---|---:|---:|---:|---:|---:|
| Full OAP | 38.08 | 7.50 | 20.05 | 23.69 | 27.13 |
| No complete-to-partial: two partial views | 38.38 | 9.68 | 29.24 | 28.70 | 30.72 |
| No part loss | 34.65 | 6.06 | 11.70 | 15.27 | 20.73 |
| No temporal loss | 39.44 | 7.02 | 15.53 | 21.15 | 25.81 |
| No global loss | 39.21 | 8.08 | 14.94 | 19.84 | 29.49 |
| No CE | 39.29 | 7.26 | 22.52 | 25.99 | 27.42 |
| No visibility gating | 39.56 | 7.82 | 16.44 | 22.95 | 27.50 |
| No part masking, main views only | 41.19 | 6.11 | 18.34 | 23.14 | 29.55 |

Two-partial training improves every severity-0.5 corruption family in every seed relative to full OAP; its random-joint gain averages **9.19 pp**. This challenges the intended pairing claim. The implementation still uses a near-complete temporal crop in this ablation, so this result specifically compares the main training views. It is also confounded by the repeated confidence multiplication and different augmentation compositions described below.

The part-loss ablation provides the most consistent positive component signal: full OAP beats `no_part` on random, dynamic, and temporal severity 0.5 in all five seeds. However, `no_part` leaves the part projector untrained. Matching on that random projector confounds loss benefit with whether the retrieval head received training. Keep the global-only comparison and add a trained part-head control before claiming isolated superiority of the part contrastive formulation.

Visibility gating is not supported by a clear across-the-board gain. Turning it off changes the objective scale as well as pair weighting. A scale-matched ablation is needed to isolate gating. Training-loss magnitudes alone do not resolve this: the gated part loss falls as masking increases, even without improved representations. The logs lack validation accuracy, training accuracy, reliability distributions, and gradient diagnostics, so they cannot distinguish underfitting from loss interference reliably.

The ST-GCN pilot has 364,057 parameters versus 100,938 for the TCN at the same class count. It reaches official NM/BG/CL **53.69/35.04/20.46** using the global descriptor. Adding parts reduces its pooled clean score from 59.67 to 57.85 and its body-part 0.5 score from 19.72 to 13.27. Thus, equal-weight part concatenation is not consistently beneficial across backbones. This is a one-seed architecture/capacity signal, not evidence that OAP beats ST-GCN CE or masked CE.

For external context, the official FastPoseGait model zoo reports HRNet CASIA-B vanilla GaitGraph2 at 80.29/71.40/63.80 and GPGait at 93.60/80.15/69.29. These are external reference results, not local reproductions. Even the ST-GCN pilot remains well below them on the official protocol. Reproducing a reference recipe would make backbone/input limitations much easier to distinguish from objective limitations. [Official model zoo](https://github.com/BNU-IVC/FastPoseGait/blob/main/docs/model_zoo.md).

**Confirmed implementation and experiment-design issues, in priority order**

1. **High priority: confidence still changes geometry during augmentation and evaluation.** `augment.py:100`, `:113`, `:175`, and `:226` multiply coordinates by a continuous visibility/confidence value. `pose.weight_coordinates: false` only changes normalization in `data.py`; it does not change those four paths. A diagnostic tensor with every coordinate equal to 1 and confidence 0.8 becomes 0.8 after each nominally no-op transform. `corrupt(..., severity=1e-10)` removed zero joints yet changed all coordinates from 1 to 0.8; severity zero returned 1. The real source mean confidence is 0.8521. With constant confidence c, generic SupCon's crop-plus-generic path applies roughly c², and OAP's generic-plus-speed-plus-corruption strong path applies roughly c³, before accounting for interpolation and noise. This creates different geometry across methods and views. The clean/positive-severity evaluation boundary changes geometry even when almost nothing is occluded. Separate binary presence from continuous confidence; use presence or a newly sampled binary keep mask to zero coordinates, and carry confidence separately for weighting. Preserve observed coordinates under no-op transforms. Reevaluate with corrected semantics and retrain the controlled matrix; improvement is a hypothesis until measured.

2. **High priority: `no_part_masking` is not a full body-part-mask ablation.** `training_views` restricts the main branch to random-joint/temporal masking, but `experiment.py:161` calls `occlusion_view` with default families for the temporal branch. The one-batch diagnostic recorded `('random_joint', 'temporal')` followed by `('random_joint', 'body_part', 'temporal')`. Route one explicit corruption policy through every branch. For `no_complete_partial`, either also change the temporal pairing or label the ablation as applying only to the main views.

3. **High priority: aggregation combines incompatible experiments.** `reporting.py:94` groups by dataset, method, descriptor, protocol, corruption, severity; it omits backbone, configuration identity, dataset checksum and seed uniqueness. Running aggregation on the current folder would count **11 OAP runs**, combining ten TCN executions of five seeds with one ST-GCN run. The existing root summary has only the original five-seed OAP core results and is stale relative to the 70 available runs. Introduce an explicit experiment/cohort ID and a hash of resolved scientific configuration, retaining backbone, data/protocol checksum, representation and effective epochs. Within each group require unique training seeds or an explicitly modeled replicate index. Do not overwrite the current summaries until grouping is fixed.

4. **High priority for scientific interpretation: there is no validation identity split.** The actual data have train/gallery/probe only. `_train` uses a fixed epoch count, and `run_sensitivity.py` calls the same test-evaluating runner. There are no sensitivity result files locally, so this is a workflow risk rather than proof that the sensitivity sweep was run. Hold out a predeclared subset of the 74 training identities for development gallery/probe evaluation. Add explicit validation protocol roles; a generic unused `val` label is insufficient. Freeze the recipe, then train on all 74 before the final 50-identity evaluation. Because these test results have already been examined repeatedly, be transparent about exploratory development and reserve a separate dataset for independent confirmation.

5. **Likely performance bottleneck: retrieval ignores part reliability.** `_descriptors` concatenates one normalized global and five normalized part vectors with equal block weights. A completely absent part still passes a zero pooled feature through biased layers and becomes a unit-norm vector. The diagnostic confirmed reliability 0 with descriptor norm 1. Such a part contributes the same nominal score weight as an observed part, and all five parts collectively receive five-sixths of the concatenated dot product. Test a separate global/part mixture coefficient and pairwise visibility-overlap weighting, normalized by the weight of usable parts. Use global-only fallback when no parts overlap. Choose weights on validation, and retain global-only results. This is a motivated proposed experiment, not a measured improvement.

6. **Medium priority: nominal corruption levels are not comparable missing fractions.** Exact enumeration of the COCO part sampler gives mean masked fractions 18.82%, 35.29%, 49.41%, and 70.59% at nominal 0.1/0.3/0.5/0.7. The five anatomical regions exclude all five face joints; severity 0.7 removes all 12 represented body joints, leaving only the head. Interior dynamic-joint masking has expected rate 3p²−2p³, namely 2.8%, 21.6%, 50%, 78.4%, plus boundary effects. Log actual missingness, and report held-out-limb/torso cases alongside severity curves. Keep nominal and measured severity explicitly distinct.

7. **Medium priority: random clip selection wraps long clips too.** `PoseDataset.__getitem__` samples a start anywhere in the real sequence and applies modulo indexing regardless of length. A 100-frame sequence with a 60-frame window crosses the end-to-start boundary for 59% of start positions. That joins unrelated phases unless the sequence is exactly periodic. Sample a genuinely contiguous window when length is at least 60, and reserve repetition for shorter sequences. Evaluate multiple fixed-length windows or track full-sequence versus clip evaluation as a validation comparison; do not assume the representation change is harmless.

Additional maintenance work: validate `val` identity separation and finite visibility; include views/conditions in the data checksum; save training checkpoints and history before the long evaluation stage so evaluation failures do not discard training; record dirty-tree/source hashes, effective corruption-realization overrides, GPU details and deterministic settings. Use separate RNG streams for main views, temporal views and evaluation masks, so removing a loss does not silently alter unrelated random draws. Audit BatchNorm exposure when comparing two-forward-pass methods with methods using two additional temporal forwards. Update README, scope and fix notes: data, CASIA-B ablations, external repositories and a ST-GCN result are present locally, contrary to several current statements.

**Recommended next experiments**

| Order | Concrete work | Decision it enables |
|---|---|---|
| 1 | Fix confidence/presence semantics, full-branch ablation policies, long-clip sampling and experiment grouping. Add focused regression checks with fractional confidence, no-op transforms, fully absent parts and duplicate seeds. Establish validation identities. | Whether subsequent comparisons measure the intended intervention. |
| 2 | On validation, reproduce CE and an augmentation-matched CE baseline with ST-GCN. Compare masked CE, global SupCon, global+part, and full OAP using the same input, sampling and backbone. Start with three seeds for development. | Whether the objective adds value beyond the stronger encoder and augmentation alone. |
| 3 | Compare complete-to-partial, two-partial and a mixture under identical geometry rules, with an explicit clean view where needed to preserve clean recognition. Use the same mask-family distribution and paired evaluation seeds. | Whether the clean/robustness trade-off comes from pairing or from unequal exposure to occlusion. |
| 4 | Test reliability-aware part matching and a separately tunable global/part mixture. Include global-only, raw pooled-part and trained part-head controls. | Whether absent-part scores and untrained-head controls explain the part-descriptor behavior. |
| 5 | Reproduce an official FastPoseGait GaitGraph/GPGait recipe; then add OAP incrementally. Compare joint/bone/velocity inputs, learning-rate schedules, identity/view-diverse batches and temperature on validation. | Whether the remaining gap is representation, optimization or the objective. |
| 6 | Freeze the recipe; run five-seed CASIA-B comparisons and the preregistered SUSTech1K real-occlusion endpoint. | Whether the corrected finding replicates beyond development and synthetic missing joints. |

The local FastPoseGait GaitGraph2 recipe uses 60-frame inputs, explicit multi-stream input construction, AdamW with OneCycleLR, a large batch, and temperature 0.01. That is a useful full reference recipe to reproduce; copying its temperature or batch size independently into this different architecture is not an evidence-based guarantee of improvement. Start with the smallest controlled changes that answer the questions above. [Official framework](https://github.com/BNU-IVC/FastPoseGait).

The supported conclusion today is narrower than the intended paper claim: **part-aware descriptors improve several synthetic-occlusion outcomes within this TCN pipeline, while the intended complete-to-partial pairing, clean-performance preservation, and reliability-gating advantage are not established.** The immediate research priority is trustworthy input semantics and a stronger controlled backbone, followed by real-occlusion validation.
