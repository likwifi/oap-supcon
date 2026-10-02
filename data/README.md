# Dataset staging area

The repository stores **folder structure, schemas, split metadata, and preparation code—not restricted benchmark data**. Put large data on a separate volume or cluster filesystem and set `OAP_DATA_ROOT=/path/to/oap_data`. Each dataset must end at:

```text
$OAP_DATA_ROOT/<dataset>/processed/dataset.npz
```

The canonical, non-pickle NPZ schema is:

| Array | Shape | Meaning |
|---|---|---|
| `x` | `[N,T,J,D]` float32 | pose coordinates, with `D=2` or `D=3` |
| `visibility` | `[N,T,J]` float32 | explicit confidence/visibility in `[0,1]` |
| `labels` | `[N]` int64 | identity index |
| `split` | `[N]` string | `train`, `val`, `gallery`, `probe`, or `test` |
| `sequence_ids` | `[N]` string | immutable source sequence ID |
| `conditions` | `[N]` string | NM/BG/CL/OCC/etc. |
| `views` | `[N]` string | camera/view identifier |

All sequences in one file currently need the same `T` and `J`; pad short sequences with zero coordinates and zero visibility. Never encode a missing joint using coordinates alone. Apply official identity-disjoint splits before synthetic masking. Standardize coordinates as `(x horizontal, y vertical)` for 2D and `(x horizontal, y vertical, z depth)` for 3D so horizontal flips and vertical-axis rotations have the same meaning across datasets.

For protocols such as CASIA-B and OU-MVLP, retain the source camera identifier in `views`. Their configurations exclude identical-view gallery samples during retrieval; missing or incorrect view metadata invalidates those results.

Dataset folders:

- `casia_b_pose`: controlled clean, bag, coat, synthetic corruption, and ablations.
- `oumvlp_pose`: large cross-view validation.
- `sustech1k`: primary real-occlusion pose experiment.
- `gait3d`: in-the-wild 2D pose; keep any SMPL-derived 3D file/result separate.
- `grew_pose`: optional large in-the-wild extension.
- `ccpg`: optional clothing/covariate extension.
- `ntu_rgbd120`: reserved for optional action-transfer work; not wired to the gait retrieval runner.
- `occgait`: reserved for its silhouette release; deliberately not accepted by the pose runner.
- `smoke`: generated data for testing the pipeline only; never report it in the paper.

Raw benchmark formats and licenses differ. Obtain each dataset from its official access page in `configs/datasets/`, retain the agreement beside your private data store, transform it to the schema above, then validate it:

```bash
export PYTHONPATH="$PWD/src"
python scripts/standardize_npz.py /private/export.npz data/casia_b_pose/processed/dataset.npz
python -m oap_supcon.cli audit --dataset casia_b_pose
```

Do not redistribute source frames, silhouettes, keypoints, identities, or annotations unless the dataset license explicitly permits it.

OccGait must not be converted by silently applying an undocumented pose estimator and then described as released skeleton data. If a derived-pose study is later approved, record the estimator, checkpoint, image-space preprocessing, failures, and output checksum, and label every result as derived pose.
