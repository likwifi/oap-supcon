#!/usr/bin/env bash
#SBATCH --job-name=gait3d_final
#SBATCH --output=logs/%x_%A_%a.out
# One-time Gait3D test confirmation: methods x seeds 11..55. The default is the
# chosen model plus the CE baseline (10 tasks). Changing OAP_FINAL_METHODS
# changes the task count; submit with a matching --array.
#SBATCH --array=0-9%3
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --exclude=volta1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
set -euo pipefail
: "${OAP_FINAL_EPOCHS:?Set OAP_FINAL_EPOCHS to the epoch budget frozen after validation}"
methods="${OAP_FINAL_METHODS:-metric_baseline,ce}"
expected=$(( $(tr ',' '\n' <<< "$methods" | wc -l) * 5 ))
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "$SLURM_ARRAY_TASK_COUNT" -ne "$expected" ]]; then
  printf 'Methods %s need %d tasks; resubmit with --array=0-%d%%3.\n' "$methods" "$expected" "$((expected - 1))" >&2
  exit 2
fi
project_root="${OAP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OAP_PYTHON:-python3}"
srun "$python_bin" scripts/run_array.py --dataset gait3d \
  --preset "${OAP_GAIT3D_PRESET:-gait3d_v2_comparison}" \
  --methods "$methods" --seeds 11,22,33,44,55 \
  --final --epochs "$OAP_FINAL_EPOCHS"
