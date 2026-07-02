#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 34:00:00
#SBATCH -o logs/train_uniref_chunk_%j.out
#SBATCH -e logs/train_uniref_chunk_%j.err

# Train ONE chunk of the full-UniRef50 (<=512) corpus, resuming from the previous
# chunk's checkpoint. Chunk index is passed via the CHUNK_IDX env var (see
# run_full_uniref_chunks.sh, which chains the 7 chunks with afterok dependencies).
#
#   sbatch --export=ALL,CHUNK_IDX=0 scripts/submit_chunk.sh                       # first chunk
#   sbatch --dependency=afterok:<prev> --export=ALL,CHUNK_IDX=1 scripts/submit_chunk.sh
#
# The resume path is resolved AT RUN TIME (inside the container) by globbing the
# previous chunk's final_epoch_* dir, since with afterok that checkpoint is
# guaranteed to exist when this job starts.

set -euo pipefail

: "${CHUNK_IDX:?set CHUNK_IDX (0..6) via --export=ALL,CHUNK_IDX=<i>}"
EXP_BASE="crosscoder_l8192_k32_bs512_full_uniref"
BASE_CONFIG="training_config_full_uniref.yaml"

CODE_DIR="/dss/dsshome1/08/ga25ley2/code/crosscode"
INTERPLM_DIR="/dss/dsshome1/08/ga25ley2/code/InterPLM"
DATA_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/data"
HF_HOME_HOST="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/hf_home"
MODEL_CHECKPOINTS_DIR="/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/model_checkpoints"

MOUNTS="${CODE_DIR}:/workspace/crosscode,${DATA_DIR}:/workspace/data,${INTERPLM_DIR}:/workspace/InterPLM,${HF_HOME_HOST}:/workspace/hf_home,${MODEL_CHECKPOINTS_DIR}:/workspace/model_checkpoints"

export WANDB_API_KEY=$(cat wandb/api_key)
export HF_HOME="/workspace/hf_home"
export PYTHONPATH="/workspace/crosscode"
mkdir -p logs

echo "Starting full-UniRef chunk ${CHUNK_IDX} on $(hostname) at $(date)"

# Command run INSIDE the container: build the per-chunk config, then train.
CONTAINER_CMD=$(cat <<EOF
set -euo pipefail
cd /workspace/crosscode
IDX=${CHUNK_IDX}
EXP_BASE=${EXP_BASE}
CHUNK_FASTA=/workspace/data/uniprot/release-2019_01/uniref/uniref50/chunks_512/chunk_0\${IDX}.fasta
CFG=/tmp/train_uniref_chunk\${IDX}.yaml

if [ "\${IDX}" -eq 0 ]; then
  RESUME=null
else
  PREV_DIR=/workspace/model_checkpoints/\${EXP_BASE}_chunk\$((IDX-1))
  RESUME=\$(ls -d \${PREV_DIR}/final_epoch_*_step_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
  if [ -z "\${RESUME}" ]; then echo "ERROR: no final checkpoint in \${PREV_DIR}"; exit 1; fi
  echo "Resuming chunk \${IDX} from \${RESUME}"
fi

# Write the per-chunk config (override fasta_path, experiment_name, resume_from).
uv run python - "\${CFG}" "\${CHUNK_FASTA}" "\${EXP_BASE}_chunk\${IDX}" "\${RESUME}" <<'PY'
import sys, yaml
out, fasta, exp, resume = sys.argv[1:5]
cfg = yaml.safe_load(open("${BASE_CONFIG}"))
cfg["data"]["token_sequence_loader"]["fasta_path"] = fasta
cfg["experiment_name"] = exp
cfg["train"]["resume_from"] = None if resume == "null" else resume
yaml.safe_dump(cfg, open(out, "w"))
print("wrote", out, "resume_from=", cfg["train"]["resume_from"])
PY

uv run wandb login
uv run crosscode/trainers/topk_crosscoder/run.py "\${CFG}"
EOF
)

srun --container-image="nvcr.io/nvidia/pytorch:25.12-py3" \
     --container-mounts="${MOUNTS}" \
     --container-workdir="/workspace/crosscode" \
     bash -c "${CONTAINER_CMD}"
