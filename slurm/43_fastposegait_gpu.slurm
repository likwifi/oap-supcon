#!/usr/bin/env bash
#SBATCH --job-name=fastposegait
#SBATCH --output=logs/%x_%j.out
# Train, then test, one official FastPoseGait model in a directory prepared by
# scripts/prepare_fastposegait.py. The framework reads configs/default.yaml and
# the Gait3D probe list relative to its working directory, so it runs from there.
# Submit once per model, for example:
#   sbatch --export=ALL,OAP_FPG_DIR=external_runs/fpg_gait3d_final,OAP_FPG_CONFIG=gpgait.yaml slurm/43_fastposegait_gpu.slurm
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --exclude=volta1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
set -euo pipefail
: "${OAP_FPG_DIR:?Set OAP_FPG_DIR to a directory written by scripts/prepare_fastposegait.py}"
: "${OAP_FPG_CONFIG:?Set OAP_FPG_CONFIG to a model config in that directory, e.g. gpgait.yaml}"
project_root="${OAP_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
prepared="$(cd "$project_root" && cd "$OAP_FPG_DIR" && pwd)"
for required in manifest.json configs/default.yaml "$OAP_FPG_CONFIG"; do
  if [[ ! -f "$prepared/$required" ]]; then
    printf 'Missing %s in %s; prepare the directory first.\n' "$required" "$prepared" >&2
    exit 2
  fi
done
python_bin="${OAP_FPG_PYTHON:-${OAP_PYTHON:-python3}}"
main="$project_root/external/FastPoseGait/fastposegait/main.py"
cd "$prepared"
for phase in train test; do
  srun "$python_bin" -m torch.distributed.run --nproc_per_node=1 "$main" \
    --cfgs "$OAP_FPG_CONFIG" --phase "$phase" --log_to_file
done
