#!/bin/bash
set -euo pipefail

WORKER="./run_xiaorong.sh"

if [[ ! -x "${WORKER}" ]]; then
    echo "Cannot execute ${WORKER}" >&2
    exit 1
fi

GPUS_PER_JOB="${GPUS_PER_JOB:-2}"

submit_job() {
    local epsilon="$1"
    local users="$2"
    local clip_radius="$3"
    local dp_sat_mode="$4"
    local tag="$5"

    local job_output
    job_output=$(sbatch --export=ALL,GPUS_PER_LAUNCH="${GPUS_PER_JOB}" "${WORKER}" \
        "${epsilon}" "${users}" "${clip_radius}" "${dp_sat_mode}" "${tag}")

    echo "${tag} -> ${job_output}"
}

EPSILONS=(1.0 2.0 4.0 8.0)
USERS=(60 80 100)
CLIP_RADII=(1.0 1.5 2.0)
DP_SAT_MODES=(fisher euclidean)


for mode in "${DP_SAT_MODES[@]}"; do
    submit_job 8.0 100 2.0 "${mode}" "epsilon_8.0-users_100-clip_2.0_mode-${mode}"
    submit_job 4.0 60 2.0 "${mode}" "epsilon_4.0-users_60-clip_2.0_mode-${mode}"
    submit_job 4.0 100 1.5 "${mode}" "epsilon_4.0-users_100-clip_1.5_mode-${mode}"
done

for eps in "${EPSILONS[@]}"; do
    submit_job "${eps}" 100 2.0 fisher "epsilon-${eps}_mode-fisher"
done

for user_count in "${USERS[@]}"; do
    submit_job 4.0 "${user_count}" 2.0 fisher "users-${user_count}_mode-fisher"
done

for clip in "${CLIP_RADII[@]}"; do
    submit_job 4.0 100 "${clip}" fisher "clip-${clip}_mode-fisher"
done
