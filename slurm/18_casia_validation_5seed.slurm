#!/usr/bin/env bash
#SBATCH --job-name=casia_val5
#SBATCH --output=logs/%x_%A_%a.out
# CASIA-B controlled lab study, validation only: 7 methods x 5 seeds = 35 tasks
# (about 1-2 GPU-hours each). No --final stage exists for this recipe.
#SBATCH --array=0-34%3
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --exclude=volta1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail
project_root="${OAP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$project_root/scripts/run_array.py" ]]; then
  printf 'Repository not found at %s; submit from the repository root or set OAP_PROJECT_ROOT.\n' "$project_root" >&2
  exit 2
fi
cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OAP_PYTHON:-python3}"
srun "$python_bin" scripts/run_array.py --dataset casia_b_pose --preset casia_validation_v2 \
  --methods ce,masked_ce,matched_ce,metric_baseline,metric_global_supcon,metric_part_supcon,oap_v2 \
  --seeds 11,22,33,44,55
