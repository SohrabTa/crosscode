# Training Time Estimate: 3 Million Protein Sequences (16384 Latents, No Prefetch)

Based on the profiling results from job `5485152` on an NVIDIA H100 (80GB), with prefetching disabled.

## Profiling Data Summary
- **Throughput:** 25.12 steps per second
- **Batch Size:** 128 sequences
- **Effective Sequences per Second:** 3,215.36 sequences/sec
- **Latents:** 16,384

## Run Projection (3,000,000 Sequences)

### 1. Training Duration
To process 3 million sequences with a batch size of 128 requires:
- **Total Steps:** 23,438 steps
- **Training Time:** ~15 minutes 33 seconds

### 2. Total Runtime Breakdown
| Phase | Duration |
| :--- | :--- |
| Model Loading (ProtT5-XL + 16k Crosscoder) | ~40 seconds |
| Norm Scaling Estimation (100k tokens) | ~5 seconds |
| Training (3,000,000 sequences) | ~15 minutes 33 seconds |
| W&B Sync & Cleanup | ~10 seconds |
| **Total Estimated Time** | **~16 minutes 28 seconds** |

## Scalability Notes
- **VRAM Utilization:** Peak VRAM reached **72.63 GB**, utilizing ~91% of the 80 GB available on the H100. This is consistent with the prefetch run.
- **Data Bottleneck:** Data Wait Time is **0.34ms**. While higher than the prefetch run (0.05ms), it remains extremely low and doesn't bottleneck the training at this throughput.
- **Comparison:** Deactivating prefetching and parallel harvesting has almost no impact on the overall training time for this specific configuration (16k latents, batch size 128 on H100).
