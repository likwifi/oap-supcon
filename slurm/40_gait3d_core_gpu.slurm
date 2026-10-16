#!/usr/bin/env bash
#SBATCH --job-name=gait3d_core
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-2%3
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --exclude=volta1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00

set -euo pipefail
project_root="${OAP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
python_bin="${OAP_PYTHON:-python3}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
srun "$python_bin" scripts/run_array.py --dataset gait3d \
  --preset "${OAP_GAIT3D_PRESET:-gait3d_v2_available}" \
  --methods metric_baseline --seeds 11,22,33
