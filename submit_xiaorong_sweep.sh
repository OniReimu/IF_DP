#!/bin/bash
set -euo pipefail

WORKER="./run_xiaorong.sh"

if [[ ! -x "${WORKER}" ]]; then
    echo "Cannot execute ${WORKER}" >&2
    exit 1
fi

GPUS_PER_JOB="${GPUS_PER_JOB:-2}"

if ! [[ "${GPUS_PER_JOB}" =~ ^[0-9]+$ ]] || (( GPUS_PER_JOB < 1 )); then
    echo "GPUS_PER_JOB must be a positive integer (got '${GPUS_PER_JOB}')" >&2
    exit 1
fi

RESOURCE_DIR=/scratch/project_462001050/myli/resources/lumi
SIF=${RESOURCE_DIR}/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif
ENV=${RESOURCE_DIR}/ifdp-venv/bin/activate

resolve_epochs() {
    local override="$1"
    if [[ -n "${override}" ]]; then
        echo "${override}"
    else
        echo "${IFDP_EPOCHS:-100}"
    fi
}

resolve_k_value() {
    local override="$1"
    if [[ -n "${override}" ]]; then
        echo "${override}"
    else
        echo "${IFDP_K:-2048}"
    fi
}

resolve_dp_param_count() {
    local override="$1"
    if [[ -n "${override}" ]]; then
        echo "${override}"
    else
        echo "${IFDP_DP_PARAM_COUNT:-20000}"
    fi
}

build_cuda_devices() {
    local gpu_count="$1"
    local devices=()
    local idx
    for ((idx=0; idx<gpu_count; idx++)); do
        devices+=("${idx}")
    done
    (IFS=,; echo "${devices[*]}")
}

print_command_preview() {
    local epsilon="$1"
    local users="$2"
    local clip_radius="$3"
    local dp_sat_mode="$4"
    local epochs_override="$5"
    local k_override="$6"
    local dp_param_override="$7"

    local epochs_value
    epochs_value=$(resolve_epochs "${epochs_override}")
    local k_value
    k_value=$(resolve_k_value "${k_override}")
    local dp_param_value
    dp_param_value=$(resolve_dp_param_count "${dp_param_override}")
    local cuda_devices
    cuda_devices=$(build_cuda_devices "${GPUS_PER_JOB}")

    cat <<EOF
singularity exec "${SIF}" \
    bash -lc "\$WITH_CONDA && \
        source ${ENV} && \
        python ablation.py \
            --dataset cifar10 \
            --model-type efficientnet \
            --k ${k_value} \
            --epochs ${epochs_value} \
            --target-epsilon ${epsilon} \
            --delta 1e-5 \
            --dp-param-count ${dp_param_value} \
            --clip-radius ${clip_radius} \
            --run-mia \
            --users ${users} \
            --calibration-k 200 \
            --dp-sat-mode ${dp_sat_mode} \
            --multi-gpu \
            --cuda-devices ${cuda_devices}"
EOF
}

submit_job() {
    local epsilon="$1"
    local users="$2"
    local clip_radius="$3"
    local dp_sat_mode="$4"
    local tag="$5"
    local epochs_override="${6:-}"
    local k_override="${7:-}"
    local dp_param_override="${8:-}"

    local export_args="ALL,GPUS_PER_LAUNCH=${GPUS_PER_JOB}"
    if [[ -n "${epochs_override}" ]]; then
        export_args+=",IFDP_EPOCHS=${epochs_override}"
    fi
    if [[ -n "${k_override}" ]]; then
        export_args+=",IFDP_K=${k_override}"
    fi
    if [[ -n "${dp_param_override}" ]]; then
        export_args+=",IFDP_DP_PARAM_COUNT=${dp_param_override}"
    fi

    local job_output
    job_output=$(sbatch --export="${export_args}" "${WORKER}" \
        "${epsilon}" "${users}" "${clip_radius}" "${dp_sat_mode}" "${tag}")

    echo "${tag} -> ${job_output}"
    echo "Command for ${tag}:"
    print_command_preview "${epsilon}" "${users}" "${clip_radius}" "${dp_sat_mode}" \
        "${epochs_override}" "${k_override}" "${dp_param_override}"
    echo
}

EPSILONS=(0.5 1.0 2.0 4.0 8.0)
USERS=(60 80 100 120 150)
CLIP_RADII=(1.0 2.0 3.0)
DP_SAT_MODES=(fisher euclidean)
EPOCHS_LIST=(100 200 300)
K_VALUES=(512 2048)
DP_PARAM_COUNTS=(1000 10000 20000 40000)


# for mode in "${DP_SAT_MODES[@]}"; do
#     submit_job 8.0 100 2.0 "${mode}" "epsilon_8.0-users_100-clip_2.0_mode-${mode}"
#     submit_job 4.0 60 2.0 "${mode}" "epsilon_4.0-users_60-clip_2.0_mode-${mode}"
#     submit_job 4.0 100 1.5 "${mode}" "epsilon_4.0-users_100-clip_1.5_mode-${mode}"
# done
submit_job 1.0 100 2.0 fisher epsilon-1.0_mode-fisher
# for eps in "${EPSILONS[@]}"; do
#     submit_job "${eps}" 100 2.0 fisher "epsilon-${eps}_mode-fisher"
# done

# for user_count in "${USERS[@]}"; do
#     submit_job 2.0 "${user_count}" 2.0 fisher "users-${user_count}_mode-fisher"
# done

# for clip in "${CLIP_RADII[@]}"; do
#     submit_job 2.0 100 "${clip}" fisher "clip-${clip}_mode-fisher"
# done

# for epochs in "${EPOCHS_LIST[@]}"; do
#     submit_job 4.0 100 2.0 fisher "epochs-${epochs}_mode-fisher" "${epochs}"
# done

# for k in "${K_VALUES[@]}"; do
#     submit_job 4.0 100 2.0 fisher "k-${k}_mode-fisher" "" "${k}"
# done

# for dp_count in "${DP_PARAM_COUNTS[@]}"; do
#     submit_job 4.0 100 2.0 fisher "dp-param-${dp_count}_mode-fisher" "" "" "${dp_count}"
# done
