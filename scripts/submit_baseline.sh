#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 33:00:00
#SBATCH -o logs/train_baseline%j.out
#SBATCH -e logs/train_baseline%j.err

# Define Configuration
CONFIG_FILE="training_config_baseline.yaml"

# Define Mounts
CROSSCODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"

# Mounts: Host:Container
MOUNTS="${CROSSCODE_DIR}:/workspace/crosscode,${DATA_DIR}:/workspace/data"

# Env
export WANDB_API_KEY=$(cat wandb/api_key)
export HF_HOME="/workspace/data/hf_home"
export PYTHONPATH="/workspace/crosscode"

mkdir -p logs

echo "Starting training run on $(hostname) at $(date)"
START_TIME=$(date +%s)
echo "Using configuration: ${CONFIG_FILE}"

srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/crosscode" \
     bash -c "uv run wandb login && uv run crosscode/trainers/topk_crosscoder/run.py ${CONFIG_FILE}"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))