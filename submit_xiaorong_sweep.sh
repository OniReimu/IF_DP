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
    local dp_shadow_epochs_override="$7"
    local reg_override="$8"
    local combined_steps_override="$9"
    local dp_lr_override="${10:-}"
    local rehearsal_override="${11:-}"

    local epochs_value
    epochs_value=$(resolve_epochs "${epochs_override}")
    local k_value
    k_value=$(resolve_k_value "${k_override}")
    local cuda_devices
    cuda_devices=$(build_cuda_devices "${GPUS_PER_JOB}")
    local optional_block=""
    if [[ -n "${dp_shadow_epochs_override}" ]]; then
        optional_block+="            --dp-epochs ${dp_shadow_epochs_override} \\\n"
        optional_block+="            --shadow-epochs ${dp_shadow_epochs_override} \\\n"
    fi
    if [[ -n "${reg_override}" ]]; then
        optional_block+="            --reg ${reg_override} \\\n"
    fi
    if [[ -n "${combined_steps_override}" ]]; then
        optional_block+="            --combined-steps ${combined_steps_override} \\\n"
    fi
    if [[ -n "${dp_lr_override}" ]]; then
        optional_block+="            --dp-lr ${dp_lr_override} \\\n"
    fi
    if [[ -n "${rehearsal_override}" ]]; then
        optional_block+="            --rehearsal-lambda ${rehearsal_override} \\\n"
    fi

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
            --clip-radius ${clip_radius} \
            --run-mia \
            --users ${users} \
            --calibration-k 200 \
            --dp-sat-mode ${dp_sat_mode} \
$(printf '%b' "${optional_block}")            --multi-gpu \
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
    local dp_shadow_epochs_override="${8:-}"
    local reg_override="${9:-}"
    local combined_steps_override="${10:-}"
    local dp_lr_override="${11:-}"
    local rehearsal_override="${12:-}"

    local export_args="ALL,GPUS_PER_LAUNCH=${GPUS_PER_JOB}"
    if [[ -n "${epochs_override}" ]]; then
        export_args+=",IFDP_EPOCHS=${epochs_override}"
    fi
    if [[ -n "${k_override}" ]]; then
        export_args+=",IFDP_K=${k_override}"
    fi
    if [[ -n "${dp_shadow_epochs_override}" ]]; then
        export_args+=",IFDP_DP_EPOCHS=${dp_shadow_epochs_override}"
        export_args+=",IFDP_SHADOW_EPOCHS=${dp_shadow_epochs_override}"
    fi
    if [[ -n "${reg_override}" ]]; then
        export_args+=",IFDP_REG=${reg_override}"
    fi
    if [[ -n "${combined_steps_override}" ]]; then
        export_args+=",IFDP_COMBINED_STEPS=${combined_steps_override}"
    fi
    if [[ -n "${dp_lr_override}" ]]; then
        export_args+=",IFDP_DP_LR=${dp_lr_override}"
    fi
    if [[ -n "${rehearsal_override}" ]]; then
        export_args+=",IFDP_REHEARSAL_LAMBDA=${rehearsal_override}"
    fi

    local job_output
    job_output=$(sbatch --export="${export_args}" "${WORKER}" \
        "${epsilon}" "${users}" "${clip_radius}" "${dp_sat_mode}" "${tag}")

    echo "${tag} -> ${job_output}"
    echo "Command for ${tag}:"
    print_command_preview "${epsilon}" "${users}" "${clip_radius}" "${dp_sat_mode}" \
        "${epochs_override}" "${k_override}" "${dp_shadow_epochs_override}" \
        "${reg_override}" "${combined_steps_override}" \
        "${dp_lr_override}" "${rehearsal_override}"
    echo
}

EPSILONS=(0.5 1.0 2.0 4.0 8.0)
USERS=(60 80 100 120 150)
CLIP_RADII=(1.0 2.0 3.0)
DP_SAT_MODES=(fisher euclidean)
EPOCHS_LIST=(100 200 300)
K_VALUES=(512 2048)
DP_SHADOW_EPOCHS=(1 3 5 10 20)
REG_VALUES=(1 2 5 10 20 50)
COMBINED_STEP_VALUES=(1 3 5 10 20)
DP_LR_VALUES=(0.01 0.02 0.05)
REHEARSAL_VALUES=(1 3 5)

# Baseline configuration (used as anchor for individual ablations)
BASE_EPS=0.5
BASE_USER=200
BASE_CLIP=3.0
BASE_MODE=fisher
BASE_EPOCH=100
BASE_K=512
BASE_SHADOW=10
BASE_REG=10
BASE_COMBINED=10
BASE_DP_LR=0.01
BASE_REHEARSAL=1

# echo "===== Sweeping EPSILONS ====="
# for val in "${EPSILONS[@]}"; do
#     submit_job "${val}" "${BASE_USER}" "${BASE_CLIP}" "${BASE_MODE}" \
#         "sweep-eps-${val}" "${BASE_EPOCH}" "${BASE_K}" "${BASE_SHADOW}" \
#         "${BASE_REG}" "${BASE_COMBINED}" "${BASE_DP_LR}" "${BASE_REHEARSAL}"
# done

# echo "===== Sweeping USERS ====="
# for val in "${USERS[@]}"; do
#     submit_job "${BASE_EPS}" "${val}" "${BASE_CLIP}" "${BASE_MODE}" \
#         "sweep-users-${val}" "${BASE_EPOCH}" "${BASE_K}" "${BASE_SHADOW}" \
#         "${BASE_REG}" "${BASE_COMBINED}" "${BASE_DP_LR}" "${BASE_REHEARSAL}"
# done

# echo "===== Sweeping CLIP_RADII ====="
# for val in "${CLIP_RADII[@]}"; do
#     submit_job "${BASE_EPS}" "${BASE_USER}" "${val}" "${BASE_MODE}" \
#         "sweep-clip-${val}" "${BASE_EPOCH}" "${BASE_K}" "${BASE_SHADOW}" \
#         "${BASE_REG}" "${BASE_COMBINED}" "${BASE_DP_LR}" "${BASE_REHEARSAL}"
# done

# echo "===== Sweeping DP_SAT_MODES ====="
# for val in "${DP_SAT_MODES[@]}"; do
#     submit_job "${BASE_EPS}" "${BASE_USER}" "${BASE_CLIP}" "${val}" \
#         "sweep-mode-${val}" "${BASE_EPOCH}" "${BASE_K}" "${BASE_SHADOW}" \
#         "${BASE_REG}" "${BASE_COMBINED}" "${BASE_DP_LR}" "${BASE_REHEARSAL}"
# done

# echo "===== Sweeping EPOCHS_LIST ====="
# for val in "${EPOCHS_LIST[@]}"; do
#     submit_job "${BASE_EPS}" "${BASE_USER}" "${BASE_CLIP}" "${BASE_MODE}" \
#         "sweep-epochs-${val}" "${val}" "${BASE_K}" "${BASE_SHADOW}" \
#         "${BASE_REG}" "${BASE_COMBINED}" "${BASE_DP_LR}" "${BASE_REHEARSAL}"
# done

# echo "===== Sweeping K_VALUES ====="
# for val in "${K_VALUES[@]}"; do
#     submit_job "${BASE_EPS}" "${BASE_USER}" "${BASE_CLIP}" "${BASE_MODE}" \
#         "sweep-k-${val}" "${BASE_EPOCH}" "${val}" "${BASE_SHADOW}" \
#         "${BASE_REG}" "${BASE_COMBINED}" "${BASE_DP_LR}" "${BASE_REHEARSAL}"
# done

# echo "===== Sweeping DP_SHADOW_EPOCHS ====="
# for val in "${DP_SHADOW_EPOCHS[@]}"; do
#     submit_job "${BASE_EPS}" "${BASE_USER}" "${BASE_CLIP}" "${BASE_MODE}" \
#         "sweep-shadow-${val}" "${BASE_EPOCH}" "${BASE_K}" "${val}" \
#         "${BASE_REG}" "${BASE_COMBINED}" "${BASE_DP_LR}" "${BASE_REHEARSAL}"
# done

# echo "===== Sweeping REG_VALUES ====="
# for val in "${REG_VALUES[@]}"; do
#     submit_job "${BASE_EPS}" "${BASE_USER}" "${BASE_CLIP}" "${BASE_MODE}" \
#         "sweep-reg-${val}" "${BASE_EPOCH}" "${BASE_K}" "${BASE_SHADOW}" \
#         "${val}" "${BASE_COMBINED}" "${BASE_DP_LR}" "${BASE_REHEARSAL}"
# done

# echo "===== Sweeping COMBINED_STEP_VALUES ====="
# for val in "${COMBINED_STEP_VALUES[@]}"; do
#     submit_job "${BASE_EPS}" "${BASE_USER}" "${BASE_CLIP}" "${BASE_MODE}" \
#         "sweep-combined-${val}" "${BASE_EPOCH}" "${BASE_K}" "${BASE_SHADOW}" \
#         "${BASE_REG}" "${val}" "${BASE_DP_LR}" "${BASE_REHEARSAL}"
# done

echo "===== Sweeping DP_LR_VALUES ====="
for val in "${DP_LR_VALUES[@]}"; do
    submit_job "${BASE_EPS}" "${BASE_USER}" "${BASE_CLIP}" "${BASE_MODE}" \
        "sweep-dp-lr-${val}" "${BASE_EPOCH}" "${BASE_K}" "${BASE_SHADOW}" \
        "${BASE_REG}" "${BASE_COMBINED}" "${val}" "${BASE_REHEARSAL}"
done

echo "===== Sweeping REHEARSAL_VALUES ====="
for val in "${REHEARSAL_VALUES[@]}"; do
    submit_job "${BASE_EPS}" "${BASE_USER}" "${BASE_CLIP}" "${BASE_MODE}" \
        "sweep-rehearsal-${val}" "${BASE_EPOCH}" "${BASE_K}" "${BASE_SHADOW}" \
        "${BASE_REG}" "${BASE_COMBINED}" "${BASE_DP_LR}" "${val}"
done

echo "===== All jobs submitted ====="
