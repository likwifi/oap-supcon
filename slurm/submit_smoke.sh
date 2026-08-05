#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash slurm/submit_smoke.sh [cpu|gpu]

Submits a short OAP-SupCon smoke validation without editing the existing
Slurm job files. The launcher uses the project-local smoke dataset.

Optional environment variables:
  OAP_PYTHON            Python executable (auto-detects env/ or .venv/)
  OAP_SLURM_PARTITION   Slurm partition (default: compute)
  OAP_SLURM_GRES        GPU GRES for gpu mode (default: gpu:1)
  OAP_SLURM_ACCOUNT     Slurm account, if required
  OAP_SLURM_QOS         Slurm QoS, if required
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

mode="${1:-cpu}"
if [[ "$mode" == "-h" || "$mode" == "--help" || "$mode" == "help" ]]; then
    usage
    exit 0
fi
if [[ "$mode" != "cpu" && "$mode" != "gpu" ]]; then
    usage >&2
    exit 2
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch is not available"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"

if [[ -n "${OAP_PYTHON:-}" ]]; then
    python_bin="$OAP_PYTHON"
elif [[ -x "$project_root/env/bin/python" ]]; then
    python_bin="$project_root/env/bin/python"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
    python_bin="$project_root/.venv/bin/python"
else
    die "no Python environment found; create env/ or .venv/, or set OAP_PYTHON"
fi

[[ -x "$python_bin" ]] || die "Python is not executable: $python_bin"

partition="${OAP_SLURM_PARTITION:-compute}"
gres="${OAP_SLURM_GRES:-gpu:1}"
data_root="$project_root/data"
mkdir -p "$project_root/logs" "$data_root"

common_args=(
    --partition="$partition"
    --chdir="$project_root"
    --export="ALL,OAP_PYTHON=$python_bin,OAP_DATA_ROOT=$data_root"
)

if [[ -n "${OAP_SLURM_ACCOUNT:-}" ]]; then
    common_args+=(--account="$OAP_SLURM_ACCOUNT")
fi
if [[ -n "${OAP_SLURM_QOS:-}" ]]; then
    common_args+=(--qos="$OAP_SLURM_QOS")
fi

if [[ "$mode" == "cpu" ]]; then
    sbatch \
        "${common_args[@]}" \
        --job-name=oap_smoke \
        --output='logs/oap_smoke_%j.out' \
        --cpus-per-task=4 \
        --mem=8G \
        --time=00:20:00 \
        --wrap="bash \"$project_root/slurm/00_smoke_cpu.slurm\""
else
    gpu_command='cd "$SLURM_SUBMIT_DIR" && nvidia-smi && "$OAP_PYTHON" -c "import torch; assert torch.cuda.is_available(), \"CUDA is unavailable\"; print(torch.cuda.get_device_name(0))" && OMP_NUM_THREADS=1 "$OAP_PYTHON" -m oap_supcon.cli run --dataset smoke --method oap_supcon --seed 11 --epochs 2 --device cuda --corruption-realizations 1'
    sbatch \
        "${common_args[@]}" \
        --job-name=oap_gpu_smoke \
        --output='logs/oap_gpu_smoke_%j.out' \
        --gres="$gres" \
        --cpus-per-task=4 \
        --mem=8G \
        --time=00:20:00 \
        --wrap="$gpu_command"
fi
