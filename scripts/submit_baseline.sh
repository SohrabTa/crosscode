#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 34:00:00
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/crosscode/train_baseline%j.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/crosscode/train_baseline%j.err

# Define Configuration
CONFIG_FILE="training_config_baseline.yaml"

# Define Mounts
CROSSCODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"
HF_HOME_HOST="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/hf_home"

# Mounts: Host:Container
MOUNTS="${CROSSCODE_DIR}:/workspace/crosscode,${DATA_DIR}:/workspace/data,${INTERPLM_DIR}:/workspace/InterPLM,${HF_HOME_HOST}:/workspace/hf_home"

# Env
export WANDB_API_KEY=$(cat wandb/api_key)
# One shared HuggingFace cache at the storage root. data/hf_home was a full
# duplicate of it (same model, same blobs) and was deleted 2026-08-13.
export HF_HOME="/workspace/hf_home"
export PYTHONPATH="/workspace/crosscode"


echo "Starting training run on $(hostname) at $(date)"
START_TIME=$(date +%s)
echo "Using configuration: ${CONFIG_FILE}"

srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/crosscode" \
     bash -c "uv run wandb login && uv run crosscode/trainers/topk_crosscoder/run.py ${CONFIG_FILE}"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))