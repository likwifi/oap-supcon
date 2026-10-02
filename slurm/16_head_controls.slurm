#!/usr/bin/env bash
#SBATCH --job-name=casia_p1_heads
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-5%3
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --exclude=volta1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail
project_root="${OAP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$project_root/scripts/run_head_controls_array.py" ]]; then
  printf 'Repository not found at %s; submit from the repository root or set OAP_PROJECT_ROOT.\n' "$project_root" >&2
  exit 2
fi
cd "$project_root"
mkdir -p logs
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OAP_PYTHON:-$project_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  printf 'Python environment not found at %s; set OAP_PYTHON to the project interpreter.\n' "$python_bin" >&2
  exit 2
fi
srun "$python_bin" scripts/run_head_controls_array.py \
  --results-root results_v2_factorial \
  --output-root results_head_controls \
  --seeds 11,22,33 \
  --modes distill,triplet_variance \
  --epochs 40 \
  --device cuda
