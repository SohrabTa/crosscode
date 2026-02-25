# Training Time Estimate: 3 Million Protein Sequences (Final Parameters)

Based on the profiling results from job `5492893` on an NVIDIA H100 (80GB).

## Profiling Data Summary
- **Throughput:** 47.61 steps/sec
- **Batch Size:** 256 sequences
- **Effective Sequences per Second:** 12,188.16 sequences/sec
- **Latents:** 8,192

## Run Projection (3,000,000 Sequences)

### 1. Training Duration
To process 3 million sequences with a batch size of 256 requires:
- **Total Steps:** 11,719 steps
- **Training Time:** ~4 minutes 6 seconds

### 2. Total Runtime Breakdown
| Phase | Duration |
| :--- | :--- |
| Model Loading (ProtT5-XL + 8k Crosscoder) | ~40 seconds |
| Norm Scaling Estimation (100k tokens) | ~5 seconds |
| Training (3,000,000 sequences) | ~4 minutes 6 seconds |
| W&B Sync & Cleanup | ~10 seconds |
| **Total Estimated Time** | **~5 minutes 1 second** |

## Scalability Notes
- **Blistering Speed:** The configuration runs extremely fast. It can blast through the entire 3-million sequence uniref50 dataset in about 5 minutes.
- **Optimal Hardware Use:** At 71.13 GB VRAM usage, the H100 is highly utilized. There's about 9GB headroom, allowing for slight spikes or additional auxiliary logging if ever needed, but pushing `batch_size` beyond 256 is likely to crash.
- **Harvester Stability:** Keeping `harvesting_batch_size` at 128 guarantees stability. The data wait time proved that harvesting is virtually zero-overhead and not a bottleneck.
