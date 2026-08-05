#!/usr/bin/env bash
#SBATCH --job-name=oap_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
python_bin="${OAP_PYTHON:-python3}"
"$python_bin" -m pytest -q
"$python_bin" -m oap_supcon.cli make-smoke
OMP_NUM_THREADS=1 "$python_bin" -m oap_supcon.cli run --dataset smoke --method oap_supcon --seed 11 --epochs 2 --device cpu --corruption-realizations 1
