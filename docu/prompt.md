# Prompt

## Role

Act as a Principal Research Engineer specializing in Mechanistic Interpretability and Protein Language Models. You are an expert in PyTorch, HPC environments (specifically Slurm and Enroot), and sparse autoencoder/crosscoder architectures.

## Project Context

We are training a **Crosscoder** (similar to the ["Sparse Crosscoders" architecture by Anthropic](https://transformer-circuits.pub/2024/crosscoders/index.html)) on the internal activations of the **ProtT5** protein language model. We aim to interpret these features using the InterPLM methodology, specifically its Swiss-Prot concept association and LLM-based descriptive validation pipelines.

* **Goal:** Train a sparse crosscoder to decompose the internal activations of ProtT5 encoder layers into interpretable features.
* **Key Architecture:** We are training a ModelHookpointAcausalCrosscoder using the trainer in `crosscode/trainers/topk_crosscoder/trainer.py`. At the same time, we are using a custom `PrefetchingIterator` class defined in `crosscode/activations_harvester.py`. This is designed to harvest activations from ProtT5 in a background thread while the Crosscoder trains on the GPU, maximizing throughput.
* **Script Location:** `crosscode/trainers/topk_crosscoder/run.py`

## Infrastructure & Environment

The training runs on the **LRZ (Leibniz Supercomputing Centre)** HPC cluster using **Enroot** containers and **Slurm**.

**Runtime & Package Management:**

* **uv:** We use `uv` for environment management and running all commands in this repository.
* **Commands:** Always prefix execution commands with `uv run`, e.g., `uv run crosscode/trainers/topk_crosscoder/run.py`.

**File System & Mounts:**

* **Code Root:** `/dss/dsshome1/08/ga25ley2/code/crosscode/` mounted to `/workspace/crosscode` inside the container.
* **Data Root:** `/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2` mounted to `/workspace/data` inside the container.
* **Artifacts:** The dataset, checkpoints, and `HF_HOME` reside in `/workspace/data`.

**Protein Sequence Data:**

* **Path:** `/workspace/data/dataset/uniref50_length_512_first_3M.fasta`
* **Number of protein sequences:**
  * 3 million protein sequences with sequence statistics:
    * Total Sequences: 3,000,000
    * Total Residues:  1,290,183,617
    * Mean Length:     430.06
    * Median Length:   429.00
    * Min Length:      11.00
    * Max Length:      512.00
    * Std Dev Length:  50.32

* **Path:** `/workspace/data/dataset/uniref50_length_512_random_3M.fasta`
* **Number of protein sequences:**
  * 3 million protein sequences with sequence statistics:
    * Total Sequences: 3,000,000
    * Total Residues:  607,449,726
    * Mean Length:     202.48
    * Median Length:   174.00
    * Min Length:      11.00
    * Max Length:      512.00
    * Std Dev Length:  123.04

**Container Details:**

* Base: `nvcr.io/nvidia/pytorch:25.12-py3`
* Container Name: `pytorch-25.12`
* WandB API Key location: `wandb/api_key` (relative to code root).

## Documentation Available

I have downloaded the relevant LRZ documentation regarding Enroot and Slurm handling.

* **Path:** `docu/doku.lrz.de-2026-02-09-*.md`
* **Instruction:** You must read these files to ensure the batch script complies with LRZ specific flags (e.g., clusters, partitions) and enroot hook mechanisms.

## Current Progress

1. **Image Import:** Official NVIDIA PyTorch 25.12 image imported.
2. **Container Creation:** Created `pytorch-25.12` from the squashfs file.
3. **Interactive Test:** I successfully ran an interactive session with the mounts described above, verified WandB login, and ran a test training run.
4. **Profiling & Instrumentation:** Instrumented `BaseTrainer` to log VRAM (peak ~24GB), throughput (~48 steps/s), and data wait time (~0ms). Found parameters that allow a full training run to complete in ~20 hours.
5. **Training Run**: Successfully ran a training run on an `lrz-hgx-h100-94x4` node with the following [Training Run Config](#training-run-config).
6. **InterPLM Integration:** Integrated Crosscoder models with InterPLM by adding adapter classes (`CrosscoderDictionaryWrapper`), utility scripts (`run_feature_collection.py`, `run_normalization.py`), and updating dependencies.
7. **Eval Dataset Creation:** Created the evaluation dataset for the InterPLM pipeline ensuring reproducibility with the following parameters:

* `extract_annotations.py`: `--input_uniprot_path /Users/sohrab.tawana/private/repos/sparse-crosscoders-prott5/data/uniprotkb_modern_score5_5k/proteins.tsv.gz --output_dir /Users/sohrab.tawana/private/repos/sparse-crosscoders-prott5/data/uniprotkb_modern_score5_5k/processed_annotations --n_shards 8`
* `prepare_eval_set.py`: `--valid_shard_range 0 3 --test_shard_range 4 7 --uniprot_dir /Users/sohrab.tawana/private/repos/sparse-crosscoders-prott5/data/uniprotkb_modern_score5_5k/processed_annotations`
* this filtered from 250 concepts to 141 concepts with at least 1,500 amino acids or 25 domains

8. **Embedding Generation:** Ran `InterPLM/submit.sh` containing `InterPLM/scripts/embed_annotations.py` on an `lrz-hgx-h100-94x4` node to generate ProtT5 embeddings for the 5k swissprot sequences in `/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/uniprotkb_modern_score5_5k/`.

9. **Crosscoder Feature Normalization:** Ran `InterPLM/scripts/submit_normalize.sh` containing `InterPLM/interplm/sae/normalize.py` on an `lrz-hgx-h100-94x4` node to normalize the crosscoder feature activation values between 0 and 1 using the 5k swissprot sequences in `/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2/uniprotkb_modern_score5_5k/`.

10. **InterPLM Evaluation Pipeline:** Ran `InterPLM/scripts/submit_eval.sh` containing `InterPLM/scripts/run_eval_pipeline.py` on an `lrz-hgx-h100-94x4` node. This script ran the full evaluation pipeline, including activation comparison, F1 calculation, and metric reporting for both validation and test sets.

11. **Feature Activation Collection:** Ran `InterPLM/scripts/submit_collect.sh` containing `InterPLM/scripts/collect_feature_activations.py` on an `lrz-hgx-h100-94x4` node. This identified the top activating protein sequences and computed statistics for the features in the crosscoder checkpoint.

12. **Dashboard Cache Generation:** Ran `InterPLM/scripts/submit_create_dashboard.sh` containing `InterPLM/scripts/create_dashboard.py` on an `lrz-hgx-h100-94x4` node. This generated the data cache required for interactive visualization in the InterPLM dashboard, integrating the crosscoder features with their biological concepts.

## Training Run Config

```yaml
data:
  token_sequence_loader:
    type: "fasta"
    fasta_path: "/workspace/data/dataset/uniref50_length_512_first_3M.fasta"
    max_sequence_length: 512
  activations_harvester:
    llms:
      - hf_model_name: Rostlab/prot_t5_xl_uniref50
        base_architecture_name: T5ForConditionalGeneration
    harvesting_batch_size: 64
    cache_mode: "no_cache"
  activations_shuffle_buffer_size: 400_000
crosscoder:
  n_latents: 8192
  k: 32
train:
  batch_size: 512
  num_steps: 2_519_889
  log_every_n_steps: 10
  topk_style: "batch_topk"
  dead_latents_threshold_n_examples: 5_000_000
  k_aux: 512 # heuristic model_dim / 2
  lambda_aux: 0.03125 # 1/32
experiment_name: "crosscoder_l8192_k32_bs512_full"
wandb:
  entity: "sohrab-train"
  project: "crosscoders-protT5"
base_save_dir: "/workspace/data/checkpoints"
cache_dir: "/workspace/data/hf_home/"
hookpoints: [
  "encoder.blocks.1.hook_resid_post",
  "encoder.blocks.2.hook_resid_post",
  "encoder.blocks.3.hook_resid_post",
  "encoder.blocks.4.hook_resid_post",
  "encoder.blocks.5.hook_resid_post",
  "encoder.blocks.6.hook_resid_post",
  "encoder.blocks.7.hook_resid_post",
  "encoder.blocks.8.hook_resid_post",
  "encoder.blocks.9.hook_resid_post",
  "encoder.blocks.10.hook_resid_post",
  "encoder.blocks.11.hook_resid_post",
  "encoder.blocks.12.hook_resid_post",
  "encoder.blocks.13.hook_resid_post",
  "encoder.blocks.14.hook_resid_post",
  "encoder.blocks.15.hook_resid_post",
  "encoder.blocks.16.hook_resid_post",
  "encoder.blocks.17.hook_resid_post",
  "encoder.blocks.18.hook_resid_post",
  "encoder.blocks.19.hook_resid_post",
  "encoder.blocks.20.hook_resid_post",
  "encoder.blocks.21.hook_resid_post",
  "encoder.blocks.22.hook_resid_post",
  "encoder.blocks.23.hook_resid_post",
  "encoder.blocks.24.hook_resid_post"
]
```
