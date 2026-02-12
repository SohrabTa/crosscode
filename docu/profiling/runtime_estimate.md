# Training Time Estimate: 3 Million Protein Sequences

Based on the profiling results from job `5480073` on an NVIDIA H100 (80GB).

## Profiling Data Summary
- **Throughput:** 48.3 steps per second
- **Batch Size:** 32 sequences
- **Effective Sequences per Second:** 1,545.6 sequences/sec

## Run Projection (3,000,000 Sequences)

### 1. Training Duration
To process 3 million sequences with a batch size of 32 requires:
- **Total Steps:** 93,750 steps
- **Training Time:** ~32 minutes 21 seconds

### 2. Total Runtime Breakdown
| Phase | Duration |
| :--- | :--- |
| Model Loading (ProtT5-XL) | ~40 seconds |
| Norm Scaling Estimation (100k tokens) | ~5 seconds |
| Training (3,000,000 sequences) | ~32 minutes 21 seconds |
| W&B Sync & Cleanup | ~10 seconds |
| **Total Estimated Time** | **~33 minutes 16 seconds** |

## Scalability Notes
- **VRAM Margin:** With only 23.9 GB utilized of the available 80 GB, the batch size could likely be increased to **128 or 256** to further improve throughput if the harvester can keep up.
- **Data Bottleneck:** The current `Data Wait Time` is nearly zero, meaning the `PrefetchingIterator` is perfectly masking the ProtT5 inference for harvesting.
