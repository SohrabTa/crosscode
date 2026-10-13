#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 27:00:00
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/crosscode/layerwise_sae/train_layerwise_sae_%j.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/logs/crosscode/layerwise_sae/train_layerwise_sae_%j.err

# Train the per-layer ProtT5 SAEs of roadmap PP-06a in one job, from one shared harvest.
# Config: training_config_layerwise_sae.yaml. Doc: documentation/experiments/12-per-layer-sae-comparison.md.
#
#   mkdir -p <logs>/crosscode/layerwise_sae && sbatch scripts/submit_layerwise_sae.sh
#
# Walltime: the doc estimates 16.2 to 20.6 h for 6 SAEs on chunk_00 (from the exp 08 throughput).
# 27 h is the upper estimate plus about 30%. The log prints the sustained steps/s and the peak GPU
# memory at steps 100, 1,000, 10,000 and each 100,000 ("progress:"). The data ends at step
# 2,199,187, so the run needs 2.2 times the "h per million steps" of that line.
# The container is the one of the crosscoder training (submit_chunk.sh), so TF32 is the same.

set -euo pipefail

CONFIG="${CONFIG:-training_config_layerwise_sae.yaml}"

CODE_DIR="${CODE_DIR:-/dss/dsshome1/08/ga25ley2/code/crosscode}"
INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"
HF_HOME_HOST="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/hf_home"
MODEL_CHECKPOINTS_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/model_checkpoints"

MOUNTS="${CODE_DIR}:/workspace/crosscode,${DATA_DIR}:/workspace/data,${INTERPLM_DIR}:/workspace/InterPLM,${HF_HOME_HOST}:/workspace/hf_home,${MODEL_CHECKPOINTS_DIR}:/workspace/model_checkpoints"

export WANDB_API_KEY=$(cat wandb/api_key)
export HF_HOME="/workspace/hf_home"
export PYTHONPATH="/workspace/crosscode"

echo "Starting layerwise SAE training (${CONFIG}) on $(hostname) at $(date), commit $(git -C "${CODE_DIR}" rev-parse HEAD)"

srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/crosscode" \
     bash -c "set -euo pipefail; uv run wandb login; uv run crosscode/trainers/topk_layerwise_sae/run.py ${CONFIG}"
