# OAP-SupCon paper experiments

**Paper:** *Occlusion-Aware Part-Level Supervised Contrastive Learning for Skeleton-Based Gait Recognition*

This folder is the self-contained experiment workspace for the occlusion-aware pose/skeleton gait paper. It separates licensed data from code, freezes the same-backbone comparisons, and writes every result to machine-readable artifacts.

## What is included

- Dataset folders and canonical NPZ schema for CASIA-B Pose, OUMVLP-Pose, SUSTech1K, Gait3D, GREW-Pose, CCPG, NTU RGB+D 120, OccGait, and a generated smoke dataset.
- A shared pose encoder accepting `D=2` or `D=3` plus an explicit visibility channel.
- CE, masked CE, generic SupCon, global occlusion SupCon, and full OAP-SupCon controls.
- Ablations for complete-to-partial pairing, global loss, part loss, temporal loss, CE, and visibility weighting.
- Independent joint-frame, dynamic-joint, anatomical-part, and contiguous-temporal corruption at the five preregistered severities.
- Crop-to-crop temporal InfoNCE, pairwise reliability-gated part contrast, and a linear masking curriculum.
- Rank-1/5/10, mAP, EER, TAR@FAR, per-condition/per-view evaluation, clean- and occluded-gallery protocols, per-severity robustness drop, identity-level paired bootstrap, audit reports, immutable run configs, checkpoints, and CSV/LaTeX aggregation.
- SLURM arrays for each long dataset workload, CASIA-B ablations, sensitivity analysis, efficiency timing, and final aggregation.
- Placeholders and a fetch script for official external baseline frameworks.

The actual benchmark data are **not** included. They require separate agreements and are far larger than this laptop can safely store. The current machine has 8 GB RAM and only about 7.8 GB free; use an external/cloud data root.

## Folder map

```text
oap_supcon_experiments/
├── configs/              base, dataset, method, and joint definitions
├── data/                 one raw/processed folder per benchmark
├── external/             official baseline repositories (fetched on demand)
├── scripts/              standardization, arrays, bootstrap, and efficiency
├── slurm/                one job file per long workload
├── src/oap_supcon/       model, losses, corruption, training, evaluation
├── tests/                leakage, masking, and loss unit tests
├── audits/               generated structural data audits
├── results/              generated run artifacts and combined tables
└── logs/                 SLURM output
```

## Install and verify locally

Run from this folder:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[test]'
export PYTHONPATH="$PWD/src"
```

The safe local validation sequence is:

```bash
make test
make smoke
make audit
make aggregate
```

The smoke data and scores are only software checks and must never appear as empirical paper results.
See `EMPIRICAL_VALIDATION.md` for the evidence gates that must pass before stating that the method works.

## Stage a real dataset

1. Obtain the benchmark through the official access URL in `configs/datasets/<name>.yaml`.
2. Create its official identity-disjoint train/validation/gallery/probe split.
3. Convert sequences to the schema in `data/README.md`, preserving pose source, condition, view, and explicit visibility.
4. Store the result locally under `data/<name>/processed/dataset.npz`, or preferably on a large volume:

```bash
export OAP_DATA_ROOT=/cluster/datasets/oap_pose
python scripts/standardize_npz.py /private/export.npz \
  "$OAP_DATA_ROOT/casia_b_pose/processed/dataset.npz"
python -m oap_supcon.cli audit --dataset casia_b_pose
```

The generic validator intentionally does not guess each provider's raw format. A dataset-specific exporter must be reviewed against the official protocol before results are valid. Keep 2D pose, lifted 3D, and SMPL-derived 3D in separate files and result tables.

For the FastPoseGait/ScienceDB CASIA-B HRNet release, use the reviewed exporter:

```bash
python scripts/convert_casia_b_hrnet.py /private/CASIA-B_HRNet \
  "$OAP_DATA_ROOT/casia_b_pose/processed/dataset.npz"
python -m oap_supcon.cli audit --dataset casia_b_pose
```

## Run one experiment

```bash
python -m oap_supcon.cli run \
  --dataset casia_b_pose \
  --method oap_supcon \
  --seed 11 \
  --device cuda
```

Valid gait-runner datasets are `smoke`, `casia_b_pose`, `oumvlp_pose`, `sustech1k`, `gait3d`, `grew_pose`, and `ccpg`. Method names are in `configs/methods.yaml`. NTU and OccGait have folders/manifests but are deliberately not accepted by this runner; see `EXPERIMENT_SCOPE.md`. To run one full controlled matrix without SLURM:

```bash
for task_id in $(seq 0 24); do
  python scripts/run_array.py \
    --dataset casia_b_pose \
    --methods ce,masked_ce,generic_supcon,global_occlusion_supcon,oap_supcon \
    --seeds 11,22,33,44,55 \
    --task-id "$task_id"
done
```

Do not run that matrix on this MacBook. It is shown for reproducibility and for non-SLURM GPU machines.

## Cloud/SLURM commands

After installing the environment and staging data on the cluster:

```bash
export OAP_PYTHON="$PWD/.venv/bin/python"
export OAP_DATA_ROOT=/cluster/datasets/oap_pose

sbatch slurm/00_smoke_cpu.slurm
sbatch slurm/10_casia_core_gpu.slurm
sbatch slurm/11_casia_ablations_gpu.slurm
sbatch slurm/20_oumvlp_core_gpu.slurm
sbatch slurm/30_sustech1k_core_gpu.slurm
sbatch slurm/40_gait3d_core_gpu.slurm
sbatch slurm/50_grew_extension_gpu.slurm
sbatch slurm/51_ccpg_extension_gpu.slurm
sbatch slurm/60_sensitivity_gpu.slurm
```

Edit the partition, account/QoS, wall time, GPU directive, and environment initialization for the target cluster. The arrays are throttled to 3–4 simultaneous jobs and request 32–64 GB RAM. See `slurm/README.md`.

## Gather results

Every successful run creates:

```text
results/<run_id>/
├── config.yaml
├── checkpoint.pt
├── history.json
├── metrics.json
└── probe_outcomes.npz
```

Regenerate combined result files without copying terminal values:

```bash
python -m oap_supcon.cli aggregate
```

This writes `results/runs.csv`, `results/summary.csv`, and `results/clean_rank1.tex`.

Compute the preregistered identity-level paired 95% CI (replace the condition label with the exact standardized SUSTech1K label):

```bash
python scripts/paired_bootstrap.py \
  results/<generic_supcon_run> \
  results/<oap_supcon_run> \
  --key official_condition_OCC \
  --samples 10000
```

Measure inference after training:

```bash
python scripts/measure_efficiency.py results/<run_id>/checkpoint.pt \
  --device cuda --warmup 100 --iterations 1000
```

Or on SLURM:

```bash
sbatch --export=ALL,CHECKPOINT=/absolute/path/to/checkpoint.pt \
  slurm/70_efficiency_gpu.slurm
sbatch slurm/90_aggregate_cpu.slurm
```

## External state-of-the-art baselines

On the cloud checkout:

```bash
bash scripts/fetch_external_baselines.sh
```

This fetches FastPoseGait, OpenGait, and GaitGraph2. Their official environment/config commands remain framework-specific. Record exact commits and label published versus locally reproduced numbers; do not force unsupported dataset/method combinations into the table.

## Before any number enters the paper

- Confirm the official split, gallery/probe conditions, pose estimator, joint map, and representation.
- Inspect rendered clean/corrupted skeletons; the automated audit is structural, not visual.
- Reproduce at least one official baseline within the preregistered tolerance.
- Freeze configs and seeds before final test evaluation.
- Use at least five CASIA-B seeds and three seeds for larger datasets.
- Treat SUSTech1K normal-gallery/occlusion-probe Rank-1 as the primary endpoint.
- Run both clean-gallery and occluded-gallery protocols and preserve camera/view metadata needed by official exclusions.
- Complete the claim gates in `EMPIRICAL_VALIDATION.md`; executable code alone is not empirical evidence.
- Report clean and corrupted performance together, including negative findings.
- Never treat the silhouette-only OccGait release as direct skeleton validation.

## Known boundaries

This is an executable reference implementation for testing the paper's objective and controls. Dataset-provider-specific raw exporters and official external-baseline configurations cannot be finalized until the licensed data and chosen pose annotations are available. The lightweight shared encoder is suitable for causal same-backbone comparisons; external SOTA claims must use their official code/configs from `external/`.
