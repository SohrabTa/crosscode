# Role
Act as a Principal Research Engineer specializing in Mechanistic Interpretability and Protein Language Models. You are an expert in PyTorch, HPC environments (specifically Slurm and Enroot), and sparse autoencoder/crosscoder architectures.

# Project Context
We are training a **Crosscoder** (similar to the ["Sparse Crosscoders" architecture by Anthropic](https://transformer-circuits.pub/2024/crosscoders/index.html)) on the internal activations of the **ProtT5** protein language model. We aim to interpret these features later using methods from the ["InterPLM" paper](https://arxiv.org/pdf/2412.12101).

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

# Immediate Tasks

### 1. Instrumentation & Profiling Code
Modify the training script (`run.py`) or `activations_harvester.py` to enable a robust **Profiling Run**. We need to measure real-world performance, not theoretical estimates.
*   **Configuration:** Use the config in `crosscode/trainers/topk_crosscoder/config.py`. Crosscoder latent dimension is **8192**. Limit the dataset to exactly **1024 protein sequences**.
*   **Memory Profiling:** Implement logging to measure the *peak* VRAM usage.
    *   *Critical:* Since ProtT5 and the Crosscoder share the same GPU, we need the total footprint. Use `torch.cuda.max_memory_allocated()`.
    *   Calculate and log the model size (in MB and number of parameters) of the Crosscoder itself.
*   **Throughput Profiling:** Measure wall-clock time for the total run and `steps_per_second`.
*   **Concurrency Validation:** Validate that `PrefetchingIterator` is actually working in parallel.
    *   Add a timer to measure "Data Wait Time" (how long the training loop waits for the `next(iterator)`).
    *   If parallel harvesting is working, Data Wait Time should be near zero.

### 2. Slurm Batch Script Generation
Create a production-ready `submit.sh` Slurm script to execute this profiling run.
*   **Resource Allocation:** Request 1 GPU from the `lrz-hgx-h100-94x4` partition.
*   **Enroot Execution:** Use the provided documentation to determine the correct way to invoke Enroot non-interactively (e.g., `srun --container-image`).
*   **Environment:** Set `HF_HOME` to `/workspace/data`, load `WANDB_API_KEY` from `wandb/api_key`, and ensure python path is correct.

### 3. Execution & Reporting Plan
*   Run the profiling job.
*   Analyze the logs to answer:
    1.  Does the total VRAM usage fit comfortably within the H100 limits, or are we close to OOM?
    2.  Is the `PrefetchingIterator` effectively masking the harvesting latency?
    3.  What is the projected time for a full training run based on these 1024 samples?

# Output Requirements
1.  Summarize findings from the LRZ documentation on how to run Enroot batch jobs.
2.  Provide the **code modifications** in `run.py` and `activations_harvester.py` to log VRAM, Model Size, and Data Wait Time.
3.  Provide the content of `submit.sh`.