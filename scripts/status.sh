#!/usr/bin/env bash
# Show scheduler and artifact progress for OAP-SupCon benchmark arrays.
#
# Usage:
#   bash scripts/status.sh                 # all current user's OAP jobs
#   bash scripts/status.sh JOB_ID          # one submitted array
#   bash scripts/status.sh JOB_ID -v       # include recent log output
#
# Optional overrides:
#   OAP_RESULTS_ROOT=results_v2
#   OAP_TARGET_RUNS=15
set -uo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

job_id="${OAP_JOB_ID:-}"
verbose=0
for arg in "$@"; do
    case "$arg" in
        -v|--verbose) verbose=1 ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                job_id="$arg"
            else
                printf 'Unknown argument: %s\n' "$arg" >&2
                exit 2
            fi
            ;;
    esac
done

results_root="${OAP_RESULTS_ROOT:-results_v2}"

job_name=""
if command -v squeue >/dev/null 2>&1; then
    if [[ -n "$job_id" ]]; then
        job_name="$(squeue -j "$job_id" -h -o '%j' 2>/dev/null | head -1)"
    else
        job_name="$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -E '^casia_v2_(dev|final)$' | head -1)"
    fi
fi
if [[ -z "$job_name" && -n "$job_id" ]] && command -v sacct >/dev/null 2>&1; then
    job_name="$(sacct -j "$job_id" -n -X -o JobName 2>/dev/null | awk 'NF {print $1; exit}')"
fi

target="${OAP_TARGET_RUNS:-}"
if [[ -z "$target" ]]; then
    case "$job_name" in
        casia_v2_dev) target=15 ;;
        casia_v2_final) target=5 ;;
        *) target='?' ;;
    esac
fi

done_n=$(find "$results_root" -name metrics.json 2>/dev/null | wc -l | tr -d ' ')
fail_n=$(find "$results_root" -name failure.json 2>/dev/null | wc -l | tr -d ' ')
start_n=$(find "$results_root" -mindepth 1 -maxdepth 1 -type d -name 'casia_b_pose_*' 2>/dev/null | wc -l | tr -d ' ')

run_n=0
pend_n=0
if command -v squeue >/dev/null 2>&1; then
    if [[ -n "$job_id" ]]; then
        queue_states="$(squeue -j "$job_id" -h -r -o '%T' 2>/dev/null || true)"
    else
        queue_states="$(squeue -u "$USER" -h -r -n casia_v2_dev,casia_v2_final -o '%T' 2>/dev/null || true)"
    fi
    run_n=$(awk '$1 == "RUNNING" {n++} END {print n+0}' <<<"$queue_states")
    pend_n=$(awk '$1 == "PENDING" {n++} END {print n+0}' <<<"$queue_states")
fi

printf 'job %s%s | artifacts: completed %s/%s, failed %s, started %s | queue: running %s, pending %s\n' \
    "${job_id:-all}" "${job_name:+ ($job_name)}" "$done_n" "$target" "$fail_n" "$start_n" "$run_n" "$pend_n"

if [[ -n "$job_id" ]] && command -v sacct >/dev/null 2>&1; then
    printf '\nSLURM ACCOUNTING:\n'
    sacct -j "$job_id" -X -n -P -o JobIDRaw,State,Elapsed,ExitCode 2>/dev/null \
        | awk -F'|' -v prefix="${job_id}_" \
            '$1 ~ ("^" prefix "[0-9]+$") {count[$2]++} END {for (state in count) printf "  %-20s %d\n", state, count[state]}' \
        | sort || true
fi

if [[ "$fail_n" -gt 0 ]]; then
    printf '\nFAILED RUNS:\n'
    while IFS= read -r failure_file; do
        run_dir=$(dirname "$failure_file")
        printf '  %s\n    %s\n' "$(basename "$run_dir")" \
            "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("message", "?").splitlines()[0])' "$failure_file" 2>/dev/null || printf '?')"
    done < <(find "$results_root" -name failure.json 2>/dev/null | sort)
fi

if [[ "$done_n" -gt 0 ]]; then
    printf '\nCOMPLETED BY METHOD:\n'
    find "$results_root" -name metrics.json -exec python3 -c \
        'import json,sys; m=json.load(open(sys.argv[1])); print(m.get("method", "?"), m.get("seed", "?"))' {} \; 2>/dev/null \
        | sort -u | awk '{seeds[$1]=seeds[$1]" "$2} END {for (method in seeds) printf "  %-26s seeds:%s\n", method, seeds[method]}' \
        | sort
fi

if [[ "$verbose" -eq 1 ]]; then
    printf '\nQUEUE:\n'
    if [[ -n "$job_id" ]]; then
        squeue -j "$job_id" -r -o '  %.18i %.18j %.10T %.10M %.24R' 2>/dev/null || true
        latest=$(find logs -maxdepth 1 -type f -name "*_${job_id}_*.out" -exec ls -t {} + 2>/dev/null | head -1)
    else
        squeue -u "$USER" -r -n casia_v2_dev,casia_v2_final -o '  %.18i %.18j %.10T %.10M %.24R' 2>/dev/null || true
        latest=$(find logs -maxdepth 1 -type f -name 'casia_v2_*.out' -exec ls -t {} + 2>/dev/null | head -1)
    fi
    if [[ -n "${latest:-}" ]]; then
        printf '\nLATEST LOG (%s):\n' "$latest"
        tail -20 "$latest" | sed 's/^/  /'
    fi
fi
