#!/bin/bash
#SBATCH --account=project_462001050  # project account to bill 
#SBATCH --partition=standard-g         # other options are small-g and standard-g
#SBATCH --gpus-per-node=4            # Use all available GPUs on the node
#SBATCH --ntasks-per-node=1          # Single task controlling multiple GPUs
#SBATCH --cpus-per-task=28           # Allocate full CPU socket per node for 8 GPUs
#SBATCH --mem-per-gpu=60G            # CPU RAM per GPU (GPU memory is always 64GB per GPU)
#SBATCH --time=8:00:00               # time limit

# this module facilitates the use of singularity containers on LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

RESOURCE_DIR=/scratch/project_462001050/myli/resources/lumi
SIF=${RESOURCE_DIR}/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif
ENV=${RESOURCE_DIR}/ifdp-venv/bin/activate


singularity exec "${SIF}" \
    bash -lc  "\$WITH_CONDA  &&\
        source ${ENV} &&\
        python -m debugpy --listen 0.0.0.0:5678 --wait-for-client  ablation.py \
            --model-type vit \
            --dataset cifar10 \
            --k 512 \
            --epochs 100 \
            --target-epsilon 0.5 \
            --delta 1e-5 \
            --dp-layer module.backbone.heads \
            --clip-radius 2.0 \
            --run-mia \
            --users 200 \
            --calibration-k 400 \
            --dp-sat-mode fisher \
            --dp-epochs 10 \
            --shadow-epochs 10 \
            --dp-lr 1e-2 \
            --public-pretrain-exclude-classes 0,1 \
            --non-iid \
            --full-complement-noise \
            --reg 10\
            --combined-steps 10\
            --multi-gpu\
            --cuda-devices 0,1,2,3"
