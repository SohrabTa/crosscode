# Walkthrough: Profiling 16k Latent Crosscoder Training (No Prefetch)

I have completed the profiling run for the 16384-latent Crosscoder with `use_parallel_harvesting` set to `False` and `prefetch_buffer_size` set to `0`. This run aimed to measure the impact of disabling data prefetching on an H100 (80GB).

## Changes Made

### Configuration
- **Batch Size**: `128`
- **Latents**: `16384`
- **Prefetching**: Disabled (`use_parallel_harvesting: False`, `prefetch_buffer_size: 0`).

## Profiling Results (Job 5485152)

| Metric | Value |
| :--- | :--- |
| **Model Parameters** | 805,347,328 |
| **Model Size (MB)** | 3,072.16 |
| **Peak VRAM (GB)** | 72.63 |
| **Throughput (steps/sec)** | 25.12 |
| **Effective Throughput (seq/sec)** | 3,215.36 |
| **Data Wait Time (sec)** | 0.00034 |

## Analysis
1.  **Negligible Impact of Prefetching:** Disabling prefetching resulted in a Very minor increase in `data_wait_time` (from 0.05ms to 0.34ms). However, the training throughput remained practically identical (25.12 vs 25.10 steps/sec).
2.  **Resource Utilization:** VRAM usage stayed at ~91% (72.63 GB), confirming that prefetch buffers (when active) consume very little memory compared to the model and activations.
3.  **Bottleneck Analysis:** At this scale (128 batch size, 16k latents), the GPU computation time per step is large enough to naturally mask the activation harvesting time, even without explicit prefetching.
4.  **Efficiency:** Total training time for the full 3M sequence dataset is estimated at approximately **16.5 minutes**, matching the optimized run.
