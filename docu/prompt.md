# Role
Act as a Principal Research Engineer specializing in Mechanistic Interpretability and Protein Language Models. You are an expert in PyTorch, HPC environments (specifically Slurm and Enroot), and sparse autoencoder/crosscoder architectures.

# Project Context
We are training a **Crosscoder** (similar to the ["Sparse Crosscoders" architecture by Anthropic](https://transformer-circuits.pub/2024/crosscoders/index.html)) on the internal activations of the **ProtT5** protein language model. We aim to interpret these features using the InterPLM methodology, specifically its Swiss-Prot concept association and LLM-based descriptive validation pipelines.

*   **Goal:** Train a sparse crosscoder to decompose the internal activations of ProtT5 encoder layers into interpretable features.
*   **Key Architecture:** We are training a ModelHookpointAcausalCrosscoder using the trainer in `crosscode/trainers/topk_crosscoder/trainer.py`. At the same time, we are using a custom `PrefetchingIterator` class defined in `crosscode/activations_harvester.py`. This is designed to harvest activations from ProtT5 in a background thread while the Crosscoder trains on the GPU, maximizing throughput.
*   **Script Location:** `crosscode/trainers/topk_crosscoder/run.py`

# Infrastructure & Environment
The training runs on the **LRZ (Leibniz Supercomputing Centre)** HPC cluster using **Enroot** containers and **Slurm**.

**File System & Mounts:**
*   **Code Root:** `/dss/dsshome1/08/ga25ley2/code/crosscode/` mounted to `/workspace/crosscode` inside the container.
*   **Data Root:** `/dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/ga25ley2` mounted to `/workspace/data` inside the container.
*   **Artifacts:** Data, checkpoints, and `HF_HOME` reside in `/workspace/data`.

**Container Details:**
*   Base: `nvcr.io/nvidia/pytorch:25.12-py3`
*   Container Name: `pytorch-25.12`
*   WandB API Key location: `wandb/api_key` (relative to code root).

# Documentation Available
I have downloaded the relevant LRZ documentation regarding Enroot and Slurm handling.
*   **Path:** `docu/doku.lrz.de-2026-02-09-*.md`
*   **Instruction:** You must read these files to ensure the batch script complies with LRZ specific flags (e.g., clusters, partitions) and enroot hook mechanisms.

# Current Progress
1.  **Image Import:** Official NVIDIA PyTorch 25.12 image imported.
2.  **Container Creation:** Created `pytorch-25.12` from the squashfs file.
3.  **Interactive Test:** I successfully ran an interactive session with the mounts described above, verified WandB login, and ran a test training run.
4.  **Profiling & Instrumentation:** Instrumented `BaseTrainer` to log VRAM (peak ~24GB), throughput (~48 steps/s), and data wait time (~0ms). Verified that `PrefetchingIterator` effectively masks activation harvesting latency. Full run on 3M sequences estimated at ~33 minutes.
5.  **16k Latent Profiling**: Optimized training for 16k latents on H100 (80GB) by increasing batch size to 128. Achieved ~3,213 seq/sec throughput with ~73GB peak VRAM. Full 3M sequence run estimated at ~17 minutes.

# Immediate Tasks
