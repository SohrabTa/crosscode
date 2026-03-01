# Walkthrough: Final Profiling Run

I have completed the final profiling run to optimize parameters for the Crosscoder training. This run was tailored for the NVIDIA H100 (80GB) by pushing batch sizes and the shuffle buffer.

## Changes Made

### Configuration Adjustments
- **Training Batch Size (`batch_size`)**: Increased to **256** (from 128). This allows processing more sequences per backward pass and effectively pushes the throughput.
- **Harvesting Batch Size (`harvesting_batch_size`)**: Kept at **128** after encountering an Out of Memory error at 256. 128 is stable and does not bottleneck the training.
- **Shuffle Buffer Size (`activations_shuffle_buffer_size`)**: Reverted to **400,000** for this stable base run.

## Profiling Results (Job 5492893)

| Metric | Value |
| :--- | :--- |
| **Model Parameters** | 402,685,952 |
| **Model Size (MB)** | 1,536.13 |
| **Peak VRAM (GB)** | 71.13 |
| **Throughput (steps/sec)** | 47.61 |
| **Effective Throughput (seq/sec)** | 12,188.16 |
| **Data Wait Time (sec)** | 0.00078 |

## Analysis
1.  **VRAM Utilization:** Peak VRAM reached 71.13 GB (~89% of the 80GB H100). The `batch_size` of 256 pushes the limit closely and seems to be the sweet spot for maximum sequence throughput without OOM errors on a single GPU.
2.  **Incredible Throughput Gain:** By combining the smaller latent dimension (8192) with the pushed batch size (256), the throughput skyrocketed to ~12,188 sequences per second.
3.  **Data Wait Time:** It is slightly higher than previous runs (0.78ms vs 0.05ms), but still well under a millisecond. This indicates that the activation harvester is keeping up beautifully, despite the much higher sequence consumption rate.
