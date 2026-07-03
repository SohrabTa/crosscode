#!/bin/bash
# Chain the 7 full-UniRef50 (<=512) training chunks as dependent Slurm jobs.
# Chunk 0 starts fresh; each later chunk waits for the previous to finish OK
# (afterok) and resumes from its checkpoint. Run from the crosscode repo root on
# the LRZ login node:
#
#   bash scripts/run_full_uniref_chunks.sh            # submit all 7
#   bash scripts/run_full_uniref_chunks.sh 3          # (re)start from chunk 3, resuming chunk 2
#
# Prints the job IDs. Cancel the whole chain with: scancel <first_id> ... or
# scancel -u $USER. Verify the chunks exist on the cluster first (see the doc).

set -euo pipefail

N_CHUNKS=3
START=${1:-0}
SUBMIT="scripts/submit_chunk.sh"

prev_job=""
for i in $(seq "${START}" $((N_CHUNKS-1))); do
  if [ -z "${prev_job}" ]; then
    if [ "${i}" -eq 0 ]; then
      dep=""
    else
      # Restarting mid-chain: chunk i resumes from chunk i-1's existing checkpoint,
      # no dependency (assumes chunk i-1 already finished).
      dep=""
    fi
  else
    dep="--dependency=afterok:${prev_job}"
  fi

  jid=$(sbatch ${dep} --export=ALL,CHUNK_IDX=${i} --parsable "${SUBMIT}")
  echo "submitted chunk ${i}: job ${jid} ${dep:+(after ${prev_job})}"
  prev_job="${jid}"
done

echo "All chunks submitted. Final job: ${prev_job}"
