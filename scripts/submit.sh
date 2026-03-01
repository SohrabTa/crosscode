#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

# Define Configuration
CONFIG_FILE="training_config.yaml"

# Define Mounts
CODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2"

# Mounts: Host:Container
MOUNTS="${CODE_DIR}:/workspace/crosscode,${DATA_DIR}:/workspace/data"

# Env
export WANDB_API_KEY=$(cat wandb/api_key)
export HF_HOME="/workspace/data/hf_home"
export PYTHONPATH="/workspace/crosscode"

mkdir -p logs

echo "Starting training run on $(hostname) at $(date)"
echo "Using configuration: ${CONFIG_FILE}"

srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/crosscode" \
     bash -c "uv run wandb login && uv run crosscode/trainers/topk_crosscoder/run.py ${CONFIG_FILE}"