#!/usr/bin/env bash
#SBATCH --job-name=grew_core
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-14%3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
python_bin="${OAP_PYTHON:-python3}"
srun "$python_bin" scripts/run_array.py --dataset grew_pose --methods ce,masked_ce,generic_supcon,global_occlusion_supcon,oap_supcon --seeds 11,22,33

