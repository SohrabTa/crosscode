# Walkthrough: Profiling Crosscoder Training

I have instrumented the Crosscoder training code to measure performance and resource usage. A profiling run has been submitted to the LRZ HPC cluster.

## Changes Made

### Instrumentation
I modified `base_trainer.py` to measure:
- **Peak VRAM**: Using `torch.cuda.max_memory_allocated()`.
- **Throughput**: Calculating `steps_per_second`.
- **Data Wait Time**: Measuring how long the training loop waits for the next batch of activations.
- **Model Size**: Logging the number of parameters and the total size in MB.

### Profiling Setup
- Created `profile_config.yaml` with a 8192-latent Crosscoder and parallel harvesting enabled.
- Created `submit_profile.sh` Slurm script tailored for LRZ's H100 nodes.

## Profiling Results (Job 5480073)

| Metric | Value |
| :--- | :--- |
| **Model Parameters** | 402,685,952 |
| **Model Size (MB)** | 1,536.12 |
| **Peak VRAM (GB)** | 23.90 |
| **Throughput (steps/sec)** | 48.31 |
| **Data Wait Time (sec)** | 0.00005 |

## Analysis
1.  **VRAM Usage:** The total VRAM footprint (ProtT5-XL + 8k Latent Crosscoder) is ~23.9 GB, which fits easily within the 80GB H100 limits.
2.  **Prefetching Efficiency:** Data Wait Time is extremely low (0.05ms), indicating that `PrefetchingIterator` is effectively masking activation harvesting latency.
3.  **Throughput:** At ~48 steps/sec with batch size 32, the trainer is processing ~1,545 sequences/sec.
