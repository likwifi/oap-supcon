# Experiment scope and execution status

This matrix prevents optional datasets from being mistaken for evidence required by the primary gait claim.

| Dataset/workload | Paper role | Local | SLURM | Status |
|---|---|---:|---:|---|
| Smoke | software verification only | yes | `00` | implemented and tested |
| CASIA-B Pose | controlled baselines, covariates, full ablations | audit only | `10`, `11`, `60` | runner ready; licensed data/export required |
| OUMVLP-Pose | large cross-view validation | audit only | `20` | runner ready; licensed data/export required |
| SUSTech1K | primary real-occlusion endpoint | audit only | `30` | runner ready; normal-gallery/OCC-probe split required |
| Gait3D | in-the-wild external validation | audit only | `40` | runner ready; keep 2D and SMPL 3D separate |
| GREW-Pose | optional large replication | audit only | `50` | runner ready; data/export required |
| CCPG | optional clothing extension | audit only | `51` | runner ready; data/export required |
| NTU RGB+D 120 | optional action-transfer claim | no | none | folder reserved; needs a separate official action-classification runner |
| OccGait | optional silhouette occlusion | no | none | folder reserved; direct pose run scientifically invalid because released pose is unavailable |
| External SOTA | GPGait++, GaitHeat, SkeletonGait, GaitGraph2 | no | framework-specific | fetch script ready; official configs must be selected after data access |
| Aggregation/bootstrap/efficiency | reporting and statistics | yes | `70`, `90` | implemented and tested on smoke artifacts |

## What must happen before cloud training

1. Secure each dataset agreement and official annotations.
2. Write/review a provider-specific exporter into the canonical NPZ schema.
3. Audit identity separation, condition/view counts, missingness, and checksums.
4. Render and manually inspect skeleton examples and every corruption family.
5. Reproduce an official baseline within the preregistered tolerance.
6. Freeze the configuration, training seeds, corruption seeds, and primary condition label.

The generic runner cannot certify that a raw-to-NPZ exporter follows an official protocol; that review is necessarily dataset-specific.

