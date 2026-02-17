# Training Time Estimate: 3 Million Protein Sequences (16384 Latents)

Based on the profiling results from job `5481835` on an NVIDIA H100 (80GB).

## Profiling Data Summary
- **Throughput:** 25.10 steps per second
- **Batch Size:** 128 sequences
- **Effective Sequences per Second:** 3,212.8 sequences/sec
- **Latents:** 16,384

## Run Projection (3,000,000 Sequences)

### 1. Training Duration
To process 3 million sequences with a batch size of 128 requires:
- **Total Steps:** 23,438 steps
- **Training Time:** ~15 minutes 34 seconds

### 2. Total Runtime Breakdown
| Phase | Duration |
| :--- | :--- |
| Model Loading (ProtT5-XL + 16k Crosscoder) | ~40 seconds |
| Norm Scaling Estimation (100k tokens) | ~5 seconds |
| Training (3,000,000 sequences) | ~15 minutes 34 seconds |
| W&B Sync & Cleanup | ~10 seconds |
| **Total Estimated Time** | **~16 minutes 39 seconds** |

## Scalability Notes
- **VRAM Utilization:** Peak VRAM reached **72.76 GB**, utilizing ~91% of the 80 GB available on the H100. This indicates we are near the upper bound for batch size with 16k latents on a single GPU.
- **Data Bottleneck:** Data Wait Time remains near zero (0.05ms), confirming that activation harvesting is fully masked by training even at higher throughput.
- **Efficiency:** Increasing the batch size from 32 to 128 has successfully doubled the effective sequence throughput compared to the 8192-latent run (3,213 vs 1,546 seq/sec).
