# Gait3D Released 2D Pose

Place extracted poses under `raw/<identity>/<camid_videoid>/<seq>/human_crop_f*.txt`.
`Gait3D.json` is the supplied identity/probe partition. The converter preserves
COCO-17 pixel coordinates, clips HRNet confidence to [0,1], sorts numeric frame
IDs, and stores real sequence lengths. Missing frame IDs are recorded as gaps;
frames are neither interpolated nor invented. Camera IDs are stored in `views`,
while immutable sequence IDs retain camera, video and track.

## Prepare And Audit

The staged pose archive contains fewer identities than the supplied 4,000-ID
partition. Use the available-pose export and recipe for this archive:

```bash
python3 scripts/convert_gait3d_pose.py data/gait3d/raw \
  data/gait3d/processed/dataset.npz \
  --partition data/gait3d/Gait3D.json --allow-incomplete \
  --report audits/gait3d/conversion.json
PYTHONPATH=src python3 -m oap_supcon.cli audit --dataset gait3d \
  --preset gait3d_v2_available
```

The converter refuses to overwrite a dataset. `--audit-only` checks source
coverage without creating an NPZ. Without `--allow-incomplete`, missing train
IDs, test IDs, designated probes, or probe gallery matches reject conversion.
Subset exports list every missing identity/probe and exclude only designated
probes with no available same-identity gallery. Raw files are preserved.

Both recipes read `processed/dataset.npz`. Select `gait3d_v2_available` for this
available-pose subset. For a complete partition, export without
`--allow-incomplete` and use `gait3d_v2`. The runner checks the stored protocol
against the recipe and records coverage plus the partition hash in run configs
and metrics. The filename does not determine the protocol; available-pose subset
scores are not full 4,000-ID benchmark scores.

## Compare Descriptors

```bash
PYTHONPATH=src python3 -m oap_supcon.cli run --dataset gait3d \
  --preset gait3d_v2_available --method metric_baseline --seed 11 --device cuda
sbatch slurm/40_gait3d_core_gpu.slurm
python3 scripts/analyze_gait3d.py results_gait3d_v2_available \
  --output results_gait3d_v2_available/comparison
```

The SLURM development job trains seeds 11,22,33. Each checkpoint evaluates
`global`, `raw_reliable`, and `raw_uniform`, with part fusion fixed at 0.5.
Validation holds out 300 eligible training identities using split seed 2026;
one track is the validation probe and all other tracks form its gallery.
Test identities and their official designated probes remain untouched.

After freezing the epoch budget from development, submit five final seeds:

```bash
export OAP_FINAL_EPOCHS=<epoch-budget-fixed-from-development>
sbatch slurm/41_gait3d_final_gpu.slurm
```

`--final` disables the runtime holdout, trains on all available official TRAIN_SET
identities, and evaluates the available TEST_SET sequences. Set
`OAP_GAIT3D_PRESET=gait3d_v2` for a complete dataset. Final runs go to
`results_gait3d_v2_available_final` (or `results_gait3d_v2_final` for a complete
dataset), keeping development and final stages separate. Analyze that directory
with `--seeds 11,22,33,44,55` for the final stage.

The analyzer creates ordinary run/summary CSVs, a descriptor endpoint table
(Rank-1/5/10, mAP, mINP), and hierarchical paired intervals over training seed
and probe identity. It rejects mixed data, recipes, stages, duplicate seeds,
or mismatched probe ordering. Robustness endpoints average the frozen synthetic
corruption cells; they are not annotations of real occlusion.

## Stratify By Real Pose Evidence

Gait3D has no occlusion labels. `scripts/analyze_gait3d_occlusion.py` bins the
scored probes into tertiles of HRNet low-confidence evidence and reports clean
Rank-1 per bin for every descriptor. It needs no GPU and re-runs no model. The
predeclared primary endpoint is height-matched: within each skeleton-height
tertile, the `raw_reliable` minus `global` gain among the probes with the most
joint-frames below 0.5 confidence minus that gain among those with the fewest,
averaged over the three height strata. The same interaction without height
matching is secondary; other proxies and contrasts are exploratory.

The analyzer verifies the dataset checksum and validation split against the runs.
Select the matched array-700 runs explicitly, because the same results root also
holds a duplicate seed 11 and failed attempts from array 696:

```bash
python3 scripts/analyze_gait3d_occlusion.py results_gait3d_v2_available \
  --runs gait3d_multistream_metric_baseline_seed11_20261005T123954237600Z \
         gait3d_multistream_metric_baseline_seed22_20261005T123956532607Z \
         gait3d_multistream_metric_baseline_seed33_20261005T123956517184Z \
  --output results_gait3d_v2_available/occlusion_strata
```

Outputs: `occlusion_strata.csv` (Rank-1 per bin, with per-bin frames and skeleton
height), `occlusion_contrasts.csv` (descriptor gains, occlusion effects and
high-minus-low interactions with crossed seed/identity intervals),
`probe_occlusion_proxies.csv`, and `occlusion_analysis.json`. Low confidence also
tracks low resolution: on validation, the highest tertile's median skeleton is
247 px against 346 px in the lowest, which is why the primary endpoint matches
on height.

`manual_audit_sheet.csv` lists 30 probes per primary bin in random order, with
no proxy values. Annotate visible occlusion from the released silhouettes or
skeleton renderings, then join on `item` with `manual_audit_key.csv` to check
whether the proxy tracks occlusion.

Use `--seeds 11,22,33,44,55` and the final results root for the final stage.

External pose baselines must use their official preprocessing, training and
evaluation settings with the same available identity/probe coverage. Keep 2D
pose scores separate from silhouette or SMPL-based comparisons.
