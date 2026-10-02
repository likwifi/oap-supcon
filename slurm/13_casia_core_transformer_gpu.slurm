#!/usr/bin/env bash
#SBATCH --job-name=casia_transformer
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-29%4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
python_bin="${OAP_PYTHON:-python3}"
srun "$python_bin" scripts/run_array.py --dataset casia_b_pose --backbone transformer --methods ce,masked_ce,generic_supcon,global_occlusion_supcon,oap_supcon,oap_supcon_dropout --seeds 11,22,33,44,55
