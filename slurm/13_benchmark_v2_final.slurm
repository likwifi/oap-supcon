#!/usr/bin/env bash
#SBATCH --job-name=casia_v2_final
#SBATCH --output=logs/%x_%A_%a.out
# Five confirmatory seeds for the selected metric baseline. Each run evaluates
# global, raw-reliable, and raw-uniform matching from the same checkpoint.
#SBATCH --array=0-4%3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail
: "${OAP_FINAL_EPOCHS:?Set OAP_FINAL_EPOCHS to the epoch budget frozen after validation}"
project_root="${OAP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$project_root/scripts/run_array.py" ]]; then
  printf 'Repository not found at %s; submit from the repository root or set OAP_PROJECT_ROOT.\n' "$project_root" >&2
  exit 2
fi
cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OAP_PYTHON:-python3}"
srun "$python_bin" scripts/run_array.py --dataset casia_b_pose --preset benchmark_v2 \
  --methods metric_baseline --seeds 11,22,33,44,55 \
  --final --epochs "$OAP_FINAL_EPOCHS"
