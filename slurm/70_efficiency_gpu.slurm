#!/usr/bin/env bash
#SBATCH --job-name=oap_efficiency
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
: "${CHECKPOINT:?submit with --export=ALL,CHECKPOINT=/absolute/path/to/checkpoint.pt}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
python_bin="${OAP_PYTHON:-python3}"
srun "$python_bin" scripts/measure_efficiency.py "$CHECKPOINT" --device cuda --warmup 100 --iterations 1000

