#!/bin/bash
#SBATCH --account=project_462001050
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=24:00:00

set -euo pipefail

if [[ "$#" -ne 5 ]]; then
    cat <<'USAGE' >&2
Usage: run_xiaorong.sh <epsilon> <users> <clip_radius> <dp_sat_mode> <tag>

Environment variables:
  GPUS_PER_LAUNCH   Number of GPUs to dedicate to this job (default: SLURM_GPUS_ON_NODE or 1)
USAGE
    exit 1
fi

EPSILON="$1"
USERS="$2"
CLIP_RADIUS="$3"
DP_SAT_MODE="$4"
TAG="$5"

EPOCHS="${IFDP_EPOCHS:-100}"
K_VALUE="${IFDP_K:-2048}"
DP_EPOCHS="${IFDP_DP_EPOCHS:-}"
SHADOW_EPOCHS="${IFDP_SHADOW_EPOCHS:-}"
REG_VALUE="${IFDP_REG:-}"
COMBINED_STEPS="${IFDP_COMBINED_STEPS:-}"
DP_LR="${IFDP_DP_LR:-}"
REHEARSAL_LAMBDA="${IFDP_REHEARSAL_LAMBDA:-}"

if [[ -z "${SHADOW_EPOCHS}" && -n "${DP_EPOCHS}" ]]; then
    SHADOW_EPOCHS="${DP_EPOCHS}"
fi

MAX_GPUS="${SLURM_GPUS_ON_NODE:-1}"
GPUS_PER_LAUNCH="${GPUS_PER_LAUNCH:-${MAX_GPUS}}"

if ! [[ "${GPUS_PER_LAUNCH}" =~ ^[0-9]+$ ]]; then
    echo "GPUS_PER_LAUNCH must be an integer (got '${GPUS_PER_LAUNCH}')" >&2
    exit 1
fi

if (( GPUS_PER_LAUNCH < 1 || GPUS_PER_LAUNCH > MAX_GPUS )); then
    echo "GPUS_PER_LAUNCH (${GPUS_PER_LAUNCH}) must be between 1 and ${MAX_GPUS}" >&2
    exit 1
fi

gpu_list=()
for ((idx=0; idx<GPUS_PER_LAUNCH; idx++)); do
    gpu_list+=("${idx}")
done

CUDA_SET=$(IFS=,; echo "${gpu_list[*]}")

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

RESOURCE_DIR=/scratch/project_462001050/myli/resources/lumi
SIF=${RESOURCE_DIR}/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif
ENV=${RESOURCE_DIR}/ifdp-venv/bin/activate

echo "[$(date --iso-8601=seconds)] Running ${TAG} with eps=${EPSILON}, users=${USERS}, clip=${CLIP_RADIUS}, mode=${DP_SAT_MODE}, epochs=${EPOCHS}, k=${K_VALUE}, dp-epochs=${DP_EPOCHS:-auto}, shadow-epochs=${SHADOW_EPOCHS:-3}, reg=${REG_VALUE:-50}, combined-steps=${COMBINED_STEPS:-3}, dp-lr=${DP_LR:-1e-3}, rehearsal=${REHEARSAL_LAMBDA:-0.1} on CUDA devices {${CUDA_SET}}"

python_cmd=(
    python ablation.py
    --dataset cifar10
    --model-type efficientnet
    --dp-layer backbone.classifier
    --k "${K_VALUE}"
    --epochs "${EPOCHS}"
    --target-epsilon "${EPSILON}"
    --delta 1e-5
    --clip-radius "${CLIP_RADIUS}"
    --run-mia
    --users "${USERS}"
    --calibration-k 400
    --non-iid
    --public-pretrain-exclude-classes 0,1 
    --dp-sat-mode "${DP_SAT_MODE}"
    --cuda-devices "${CUDA_SET}"
)

if [[ -n "${DP_EPOCHS}" ]]; then
    python_cmd+=("--dp-epochs" "${DP_EPOCHS}")
fi

if [[ -n "${SHADOW_EPOCHS}" ]]; then
    python_cmd+=("--shadow-epochs" "${SHADOW_EPOCHS}")
fi

if [[ -n "${REG_VALUE}" ]]; then
    python_cmd+=("--reg" "${REG_VALUE}")
fi

if [[ -n "${COMBINED_STEPS}" ]]; then
    python_cmd+=("--combined-steps" "${COMBINED_STEPS}")
fi

if [[ -n "${DP_LR}" ]]; then
    python_cmd+=("--dp-lr" "${DP_LR}")
fi

if [[ -n "${REHEARSAL_LAMBDA}" ]]; then
    python_cmd+=("--rehearsal-lambda" "${REHEARSAL_LAMBDA}")
fi

printf -v python_cmd_str '%q ' "${python_cmd[@]}"
python_cmd_str=${python_cmd_str% }

singularity exec "${SIF}" \
    bash -lc "\$WITH_CONDA && \
        source ${ENV} && \
        ${python_cmd_str}"
