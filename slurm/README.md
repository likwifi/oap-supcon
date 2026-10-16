# SLURM execution

Submit from the project root so `logs/` resolves correctly. The current cloud cluster uses partition `compute`, one CUDA GPU per task, and excludes `volta1`. Jobs `15`, `16`, `17`, `40`, and `41` include these defaults. Older GPU job templates still use partition `gpu`; override them with `sbatch --partition=compute --gres=gpu:1 --exclude=volta1 --export=ALL`. Use a Python environment that already contains `requirements.txt`, and adapt wall time, memory, account/QoS, and module/conda setup for the target cluster.

Before submission:

```bash
cd /cluster/path/oap_supcon_experiments
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
export OAP_PYTHON="$PWD/.venv/bin/python"
export OAP_DATA_ROOT=/cluster/datasets/oap_pose
```

Environment exports usually need to be placed in the submit shell, scheduler environment file, or directly in the SLURM script depending on the cluster. Each array element is one independent method/seed training run. Requeue only failed elements; completed runs have `status: complete` in `results/*/metrics.json`.

Suggested submission order:

```bash
sbatch slurm/00_smoke_cpu.slurm
sbatch slurm/10_casia_core_gpu.slurm
sbatch slurm/11_casia_ablations_gpu.slurm
sbatch slurm/20_oumvlp_core_gpu.slurm
sbatch slurm/30_sustech1k_core_gpu.slurm
sbatch slurm/40_gait3d_core_gpu.slurm
sbatch slurm/50_grew_extension_gpu.slurm       # optional
sbatch slurm/51_ccpg_extension_gpu.slurm       # optional
sbatch slurm/60_sensitivity_gpu.slurm
sbatch --export=ALL,CHECKPOINT=/path/to/checkpoint.pt slurm/70_efficiency_gpu.slurm
sbatch slurm/90_aggregate_cpu.slurm
```

After freezing one epoch budget from validation, the minimal untouched
CASIA-B confirmation is:

```bash
export OAP_FINAL_EPOCHS=<fixed-validation-derived-epoch>
sbatch slurm/13_benchmark_v2_final.slurm
```

This final array trains only the selected metric baseline for five seeds. Each
checkpoint is scored with the global, reliability-weighted raw-part, and
uniform raw-part descriptors at the fixed 0.5 local-score weight.

The arrays intentionally do not chain dependencies: baseline reproduction and dataset audits should be reviewed before expensive proposed-method jobs are accepted as paper results.

The Gait3D `40` array now runs the matched metric-baseline descriptor comparison
for three development seeds, using `gait3d_v2_available` by default. Set
`OAP_GAIT3D_PRESET=gait3d_v2` only for a complete official partition. The `41`
array runs five final seeds and requires `OAP_FINAL_EPOCHS` frozen from Gait3D
development. See `data/gait3d/README.md` for conversion and comparison reporting.
