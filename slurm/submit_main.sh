#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash slurm/submit_main.sh audit <dataset>
  bash slurm/submit_main.sh prepare-casia <absolute-CASIA-B_HRNet-directory>
  bash slurm/submit_main.sh <workflow>
  bash slurm/submit_main.sh efficiency <absolute-checkpoint-path>

Workflows:
  casia-core          CASIA-B five-method core matrix (tcn backbone)
  casia-core-stgcn    CASIA-B core matrix on the ST-GCN backbone
  casia-core-transf   CASIA-B core matrix on the skeleton-transformer backbone
  casia-dropout       oap_supcon_dropout on the tcn backbone (completes that table)
  casia-ablations     CASIA-B ablation matrix
  prepare-casia       convert and audit the ScienceDB HRNet release (CPU job)
  oumvlp-core         OUMVLP-Pose core matrix
  sustech1k-core      SUSTech1K core matrix
  gait3d-core         Gait3D core matrix
  grew-core           optional GREW-Pose extension
  ccpg-core           optional CCPG extension
  sensitivity         CASIA-B sensitivity analysis
  efficiency          checkpoint efficiency measurement
  aggregate           aggregate completed result artifacts

The launcher submits exactly one workflow per invocation. It refuses to submit
training when the expected standardized dataset.npz is absent.

Optional environment variables:
  OAP_PYTHON            Python executable (auto-detects env/ or .venv/)
  OAP_DATA_ROOT         Real-data root (default: $HOME/oap-data)
  OAP_SLURM_PARTITION   Slurm partition (default: compute)
  OAP_SLURM_GRES        GPU GRES (default: gpu:1)
  OAP_SLURM_ACCOUNT     Slurm account, if required
  OAP_SLURM_QOS         Slurm QoS, if required
  OAP_SLURM_EXCLUDE     Comma-separated nodes to keep jobs off, e.g. volta1,volta2.
                        Needed when part of the cluster has GPUs this torch build
                        has no kernels for; check torch.cuda.get_arch_list()
                        against the node's compute capability.
  OAP_SLURM_NODELIST    Restrict jobs to these nodes instead (mutually exclusive
                        with OAP_SLURM_EXCLUDE)
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
    usage
    exit 0
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

workflow="$1"
data_root="${OAP_DATA_ROOT:-$HOME/oap_supcon}"
partition="${OAP_SLURM_PARTITION:-compute}"
gres="${OAP_SLURM_GRES:-gpu:1}"
mkdir -p "$project_root/logs"

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
if [[ -n "${OAP_SLURM_EXCLUDE:-}" && -n "${OAP_SLURM_NODELIST:-}" ]]; then
    die "set only one of OAP_SLURM_EXCLUDE and OAP_SLURM_NODELIST"
fi
if [[ -n "${OAP_SLURM_EXCLUDE:-}" ]]; then
    common_args+=(--exclude="$OAP_SLURM_EXCLUDE")
fi
if [[ -n "${OAP_SLURM_NODELIST:-}" ]]; then
    common_args+=(--nodelist="$OAP_SLURM_NODELIST")
fi

dataset_folder() {
    case "$1" in
        casia_b_pose|oumvlp_pose|sustech1k|gait3d|grew_pose|ccpg) printf '%s\n' "$1" ;;
        *) die "unsupported dataset: $1" ;;
    esac
}

require_dataset() {
    local dataset="$1"
    local folder
    folder="$(dataset_folder "$dataset")"
    local path="$data_root/$folder/processed/dataset.npz"
    [[ -f "$path" ]] || die "dataset missing: $path"
}

submit_gpu_array() {
    local script="$1"
    local job_name="$2"
    local array="$3"
    local memory="$4"
    local walltime="$5"

    sbatch \
        "${common_args[@]}" \
        --job-name="$job_name" \
        --output='logs/%x_%A_%a.out' \
        --array="$array" \
        --gres="$gres" \
        --cpus-per-task=8 \
        --mem="$memory" \
        --time="$walltime" \
        --wrap="bash \"$project_root/slurm/$script\""
}

if [[ "$workflow" == "audit" ]]; then
    [[ $# -eq 2 ]] || die "audit requires one dataset name"
    dataset="$2"
    require_dataset "$dataset"
    audit_command="cd \"\$SLURM_SUBMIT_DIR\" && \"\$OAP_PYTHON\" -m oap_supcon.cli audit --dataset \"$dataset\""
    sbatch \
        "${common_args[@]}" \
        --job-name="audit_${dataset}" \
        --output='logs/%x_%j.out' \
        --cpus-per-task=4 \
        --mem=32G \
        --time=01:00:00 \
        --wrap="$audit_command"
    exit 0
fi

if [[ "$workflow" == "prepare-casia" ]]; then
    [[ $# -eq 2 ]] || die "prepare-casia requires the extracted CASIA-B_HRNet directory"
    source_dir="$2"
    [[ "$source_dir" == /* ]] || die "CASIA-B_HRNet path must be absolute"
    [[ -d "$source_dir" ]] || die "CASIA-B_HRNet directory missing: $source_dir"
    source_dir="$(cd -- "$source_dir" && pwd)"
    destination="$data_root/casia_b_pose/processed/dataset.npz"
    [[ ! -e "$destination" ]] || die "destination already exists: $destination"
    printf -v prepare_command \
        'cd %q && %q scripts/convert_casia_b_hrnet.py %q %q && %q -m oap_supcon.cli audit --dataset casia_b_pose' \
        "$project_root" "$python_bin" "$source_dir" "$destination" "$python_bin"
    sbatch \
        "${common_args[@]}" \
        --job-name=oap_prepare_casia \
        --output='logs/oap_prepare_casia_%j.out' \
        --cpus-per-task=4 \
        --mem=16G \
        --time=02:00:00 \
        --wrap="$prepare_command"
    exit 0
fi

case "$workflow" in
    casia-core)
        require_dataset casia_b_pose
        submit_gpu_array 10_casia_core_gpu.slurm casia_core '0-24%4' 32G 24:00:00
        ;;
    casia-core-stgcn)
        require_dataset casia_b_pose
        submit_gpu_array 12_casia_core_stgcn_gpu.slurm casia_stgcn '0-29%4' 32G 24:00:00
        ;;
    casia-core-transf)
        require_dataset casia_b_pose
        submit_gpu_array 13_casia_core_transformer_gpu.slurm casia_transf '0-29%4' 32G 24:00:00
        ;;
    casia-dropout)
        require_dataset casia_b_pose
        submit_gpu_array 14_casia_dropout_gpu.slurm casia_dropout '0-4%4' 32G 24:00:00
        ;;
    casia-ablations)
        require_dataset casia_b_pose
        submit_gpu_array 11_casia_ablations_gpu.slurm casia_ablate '0-39%4' 32G 24:00:00
        ;;
    oumvlp-core)
        require_dataset oumvlp_pose
        submit_gpu_array 20_oumvlp_core_gpu.slurm oumvlp_core '0-14%3' 48G 48:00:00
        ;;
    sustech1k-core)
        require_dataset sustech1k
        submit_gpu_array 30_sustech1k_core_gpu.slurm sustech_core '0-14%3' 32G 36:00:00
        ;;
    gait3d-core)
        require_dataset gait3d
        submit_gpu_array 40_gait3d_core_gpu.slurm gait3d_core '0-14%3' 48G 48:00:00
        ;;
    grew-core)
        require_dataset grew_pose
        submit_gpu_array 50_grew_extension_gpu.slurm grew_core '0-14%3' 64G 72:00:00
        ;;
    ccpg-core)
        require_dataset ccpg
        submit_gpu_array 51_ccpg_extension_gpu.slurm ccpg_core '0-14%3' 32G 36:00:00
        ;;
    sensitivity)
        require_dataset casia_b_pose
        submit_gpu_array 60_sensitivity_gpu.slurm casia_sense '0-17%3' 32G 24:00:00
        ;;
    efficiency)
        checkpoint="${2:-${CHECKPOINT:-}}"
        [[ -n "$checkpoint" ]] || die "efficiency requires an absolute checkpoint path"
        [[ "$checkpoint" == /* ]] || die "checkpoint path must be absolute"
        [[ -f "$checkpoint" ]] || die "checkpoint missing: $checkpoint"
        sbatch \
            "${common_args[@]}" \
            --job-name=oap_efficiency \
            --output='logs/oap_efficiency_%j.out' \
            --gres="$gres" \
            --cpus-per-task=4 \
            --mem=16G \
            --time=02:00:00 \
            --export="ALL,OAP_PYTHON=$python_bin,OAP_DATA_ROOT=$data_root,CHECKPOINT=$checkpoint" \
            --wrap="bash \"$project_root/slurm/70_efficiency_gpu.slurm\""
        ;;
    aggregate)
        sbatch \
            "${common_args[@]}" \
            --job-name=oap_aggregate \
            --output='logs/oap_aggregate_%j.out' \
            --cpus-per-task=4 \
            --mem=16G \
            --time=01:00:00 \
            --wrap="bash \"$project_root/slurm/90_aggregate_cpu.slurm\""
        ;;
    *)
        usage >&2
        die "unknown workflow: $workflow"
        ;;
esac
