# SLURM execution

Submit from the project root so `logs/` resolves correctly. The files assume a partition named `gpu`, one CUDA GPU, and a Python environment that already contains `requirements.txt`. Change `--partition`, `--time`, `--mem`, account/QoS, and module/conda setup to match the cloud cluster.

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

The arrays intentionally do not chain dependencies: baseline reproduction and dataset audits should be reviewed before expensive proposed-method jobs are accepted as paper results.

## Portable submission launchers

Some Slurm installations execute a private spooled copy of a submitted script.
On those systems, deriving the project directory from `BASH_SOURCE[0]` inside the
batch copy can point at Slurm's protected spool directory. The launchers below
avoid editing the workload files: they submit an official `--wrap` job that
executes the original workload file by its absolute project path.

Run CPU and GPU smoke validation first:

```bash
bash slurm/submit_smoke.sh cpu
bash slurm/submit_smoke.sh gpu
```

For real data, set the standardized dataset root, audit each dataset, and submit
one workflow at a time:

```bash
export OAP_DATA_ROOT="$HOME/oap-data"
bash slurm/submit_main.sh prepare-casia "$HOME/private-data/CASIA-B_HRNet"
# After the preparation job completes successfully:
bash slurm/submit_main.sh audit casia_b_pose
bash slurm/submit_main.sh casia-core
```

List all supported workflows and optional cluster settings with:

```bash
bash slurm/submit_main.sh --help
```
