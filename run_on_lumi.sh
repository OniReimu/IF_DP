#!/bin/bash
#SBATCH --account=project_462001050  # project account to bill 
#SBATCH --partition=standard-g         # other options are small-g and standard-g
#SBATCH --gpus-per-node=1            # Number of GPUs per node (max of 8)
#SBATCH --ntasks-per-node=1          # Use one task for one GPU
#SBATCH --cpus-per-task=7            # Use 1/8 of all available 56 CPUs on LUMI-G nodes
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
             python ablation.py --dataset wildchat --model-type qwen \
            --k 512 --epochs 200 --target-epsilon 8.0 --delta 1e-5 \
            --dp-param-count 20000 --clip-radius 2.0 \
            --run-mia --users 60 --calibration-k 200 --dp-sat-mode fisher"
