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
| `frame_counts` | `[N]` int32 | source frames before zero-padding (recommended) |

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

## CASIA-B HRNet release

The FastPoseGait/ScienceDB `CASIA-B_HRNet` release contains one `[T,17,3]`
float32 pickle per identity/sequence/view; its channels are `(x, y, HRNet
confidence)`. Convert the extracted directory with the dataset-specific,
restricted-pickle reader:

```bash
export OAP_DATA_ROOT="${OAP_DATA_ROOT:-$HOME/oap-data}"
python scripts/convert_casia_b_hrnet.py /private/CASIA-B_HRNet \
  "$OAP_DATA_ROOT/casia_b_pose/processed/dataset.npz"
python -m oap_supcon.cli audit --dataset casia_b_pose
```

The converter applies the standard 74/50 protocol: identities 001-074 are
training data; for identities 075-124, `nm-01` through `nm-04` are gallery and
`nm-05`, `nm-06`, `bg-01`, `bg-02`, `cl-01`, and `cl-02` are probes. It retains
continuous HRNet confidence as `visibility` and zero-pads shorter sequences.
Because the release contains occasional small HRNet score overshoots above
`1.0`, the exporter clips that channel to the canonical `[0,1]` range and
records the transform in the output metadata.
It validates the companion frame mappings and recognizes the release's three
known absent sequences. It refuses to overwrite an existing output.

Render representative source frames before accepting results (the renderer's
only extra dependency is `matplotlib`):

```bash
python scripts/render_casia_b_pose.py \
  /private/CASIA-B_HRNet/104/nm-01/090/090.pkl \
  audits/casia_b_pose/pose_check.png
```

OccGait must not be converted by silently applying an undocumented pose estimator and then described as released skeleton data. If a derived-pose study is later approved, record the estimator, checkpoint, image-space preprocessing, failures, and output checksum, and label every result as derived pose.
