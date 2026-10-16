#!/usr/bin/env bash
#SBATCH --job-name=descriptor_controls5
#SBATCH --output=logs/%x_%A_%a.out
# Checkpoint-only descriptor and random-head controls for each metric_baseline
# seed of a completed cohort. Defaults: the five-seed CASIA-B validation study.
# For Gait3D, set OAP_CONTROL_RESULTS=results_gait3d_v2_comparison,
# OAP_CONTROL_OUTPUT=results_descriptor_controls_gait3d, OAP_CONTROL_SEEDS=11,22,33
# and submit with --array=0-2.
#SBATCH --array=0-4%3
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --exclude=volta1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -euo pipefail
project_root="${OAP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
python_bin="${OAP_PYTHON:-python3}"
srun "$python_bin" scripts/run_descriptor_controls_array.py \
  --results-root "${OAP_CONTROL_RESULTS:-results_casia_validation_v2}" \
  --output-root "${OAP_CONTROL_OUTPUT:-results_descriptor_controls_5seed}" \
  --seeds "${OAP_CONTROL_SEEDS:-11,22,33,44,55}" \
  --device cuda
