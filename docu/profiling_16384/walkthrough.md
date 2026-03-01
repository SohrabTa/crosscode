# Walkthrough: Profiling 16k Latent Crosscoder Training

I have completed the profiling run for the 16384-latent Crosscoder. This run was optimized for the H100 (80GB) architecture by increasing the batch size and utilizing parallel activation harvesting.

## Changes Made

### Infrastructure & Configuration
- **Batch Size**: Increased to `128` (from `32` in the 8k run).
- **Latents**: Set to `16384`.
- **Visibility**: Updated `submit_profile.sh` to explicitly define and log the used configuration file.

## Profiling Results (Job 5481835)

| Metric | Value |
| :--- | :--- |
| **Model Parameters** | 805,347,328 |
| **Model Size (MB)** | 3,072.16 |
| **Peak VRAM (GB)** | 72.76 |
| **Throughput (steps/sec)** | 25.10 |
| **Effective Throughput (seq/sec)** | 3,212.8 |
| **Data Wait Time (sec)** | 0.00005 |

## Analysis
1.  **VRAM Optimization:** The 80GB H100 is now well-utilized (~91%). The 16k latent model with batch size 128 fits comfortably but leaves little room for further batch size increases.
2.  **Throughput Gain:** By quadrupling the batch size, we achieved a ~2x increase in sequence processing speed compared to the 8k latent run, even though the latent count doubled.
3.  **Harvester Performance:** The `PrefetchingIterator` continues to perform flawlessly, with data wait times remaining negligible.
4.  **Training Efficiency:** Total training time for the full 3M sequence dataset is estimated at just under **17 minutes**.
