# Empirical claim gate

The implementation can execute the paper's method, controls, ablations, and two gallery protocols. That is necessary but not sufficient evidence that OAP-SupCon works. Do not make the paper's central empirical claim until every gate below is complete on real benchmark data.

## 1. Implementation gate

- `python -m pytest -q` passes.
- A full smoke run completes for every method in `configs/methods.yaml`.
- The smoke outputs are used only for software validation and are excluded from paper tables.
- The frozen run configuration records separate global, part, and temporal temperatures; loss weights; crop ratios; severity curriculum; seed; model capacity; and dataset checksum.

## 2. Data and protocol gate

- Dataset access and redistribution terms have been reviewed.
- A provider-specific exporter has been checked against the official identity, gallery, probe, condition, and view protocol.
- Coordinates use the canonical axes documented in `data/README.md`.
- Skeleton renders confirm the joint map, left/right orientation, missingness mask, normalization, and every corruption family.
- CASIA-B and OU-MVLP evaluations exclude identical-view gallery pairs.
- Real occlusion is not silently replaced by synthetic masking or by an undocumented pose estimator.

## 3. Required controlled runs

Run all methods with the same backbone, batch construction, optimizer, epochs, and input representation.

| Workload | Methods | Seeds | Purpose |
|---|---|---:|---|
| CASIA-B Pose core | `ce`, `masked_ce`, `generic_supcon`, `global_occlusion_supcon`, `oap_supcon` | 11, 22, 33, 44, 55 | Controlled central comparison |
| CASIA-B Pose ablations | full method plus every `no_*`/unweighted configuration, including `no_part_masking` | 11, 22, 33, 44, 55 | Component attribution |
| SUSTech1K | five core methods | 11, 22, 33 | Primary real-occlusion endpoint |
| OU-MVLP Pose | five core methods | 11, 22, 33 | Large cross-view validation |
| Gait3D | five core methods | 11, 22, 33 | In-the-wild external validation |

GREW-Pose and CCPG are extensions, not substitutes for the controlled and real-occlusion endpoints. Keep 2D pose, lifted 3D, and SMPL-derived 3D as separate experiments.

## 4. Statistical gate

Report mean, standard deviation, and seed count for every result. For the pre-registered primary comparisons, use identity-level paired bootstrap confidence intervals from `probe_outcomes.npz`.

Examples:

```bash
python scripts/paired_bootstrap.py \
  results/<generic_run> results/<oap_run> \
  --key clean_gallery_body_part_0.5 --samples 10000

python scripts/paired_bootstrap.py \
  results/<generic_sustech_run> results/<oap_sustech_run> \
  --key official_condition_OCC --samples 10000
```

Record negative findings and clean/robustness trade-offs. Select checkpoints and hyperparameters using validation data only; do not tune on the final probe/test identities.

## 5. Claim criteria

The current paper defines a positive result only if all three conditions hold:

1. Full OAP-SupCon matches or improves clean Rank-1 relative to the same-backbone CE baseline.
2. Its part-occlusion robustness drop is lower at medium and severe corruption, especially at severities 0.3 and 0.5.
3. Ablations show measurable contributions from body-part-aware positive construction and the part-level contrastive loss.

The primary SUSTech1K normal-gallery/OCC-probe result must also support the synthetic-occlusion finding. If confidence intervals include no improvement, or results fail to replicate across seeds/datasets, the correct conclusion is that the claim was not supported under the tested protocol.

## 6. External validity gate

- Reproduce at least one official backbone baseline within a pre-declared tolerance.
- Record repository URLs, exact commits, environments, checkpoints, and any local configuration changes.
- Compare parameter count, inference latency, and training cost alongside recognition accuracy.
- Confine conclusions to the datasets, pose sources, populations, and occlusion conditions actually evaluated.
