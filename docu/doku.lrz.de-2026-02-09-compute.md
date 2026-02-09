This content cannot be displayed without JavaScript.
Please enable JavaScript and reload the page.

[![](/_/de.lrz.scroll.scroll-viewport-theme:scroll-webhelp-theme-lrz/1739176190507/assets/img/lrz_logo_2014_s.png)](.)

[![](/_/de.lrz.scroll.scroll-viewport-theme:scroll-webhelp-theme-lrz/1739176190507/assets/img/lrz_logo_sm.png)](.)

[![US Flag](/_/de.lrz.scroll.scroll-viewport-theme:scroll-webhelp-theme-lrz/1739176190507/assets/img/us.svg)](?showLanguage=en_US)
[![GB Flag](/_/de.lrz.scroll.scroll-viewport-theme:scroll-webhelp-theme-lrz/1739176190507/assets/img/gb.svg)](?showLanguage=en_GB)
[![DE Flag](/_/de.lrz.scroll.scroll-viewport-theme:scroll-webhelp-theme-lrz/1739176190507/assets/img/de.svg)](?showLanguage=de_DE)

* [LRZ Dokumentationsplattform](/willkommen-4537411.html)
* ...
* [High Performance Computing](/high-performance-computing-10613431.html)
* [AI Systems](/ai-systems-11484278.html)

# 2. Compute

The following table summarizes the available compute hardware and the Slurm partitions to which jobs need to be submitted. Partitions in **grey** are currently primarily dedicated to be used interactively via our **[Interactive Apps](/6-interactive-apps-10746644.html)** (notebooks) and cannot be targeted directly.

## **Partitions**

| Architecture | **Slurm Partition** | **Number of nodes** | **CPUs per node** | **CPU Memory per node** | **GPUs per node** | **Memory per GPU** |
| --- | --- | --- | --- | --- | --- | --- |
| **HGX H100 (BayernKI)** | lrz-hgx-h100-94x4 | 30 | 96 | 768 GB | 4 NVIDIA H100 | 94 GB HBM2 |
| **HGX A100** | lrz-hgx-a100-80x4 | 5 | 96 | 1 TB | 4 NVIDIA A100 | 80 GB HBM2 |
| **DGX A100** | lrz-dgx-a100-80x8 | 4 | 252 | 2 TB | 8 NVIDIA A100 | 80 GB HBM2 |
| **lrz-dgx-a100-40x8-mig** | 1 | 252 | 1 TB | 8 NVIDIA A100 | 40 GB HBM2 |
| **DGX-1 V100** | lrz-dgx-1-v100x8 | 1 | 76 | 512 GB | 8 NVIDIA Tesla V100 | 16 GB HBM2 |
| **DGX-1 P100** | lrz-dgx-1-p100x8 | 1 | 76 | 512 GB | 8 NVIDIA Tesla P100 | 16 GB HBM2 |
| **HPE Intel Skylake + P100** | lrz-hpe-p100x4 | 1 | 28 | 256 GB | 4 NVIDIA Tesla P100 | 16 GB HBM2 |
| **V100 GPU Nodes** | lrz-v100x2 (default) | 4 | 19 | 368 GB | 2 NVIDIA Tesla V100 | 16 GB HBM2 |
| **CPU Nodes** | **lrz-cpu** | 12 | 18 / 28 / 38 / 94 | min. 360 GB | -- | -- |

Run the following command on the log in node to see the partitions:

```
sinfo
```

**Quick note on the partition names:**The naming convention (e.g: lrz-hgx-h100-94x4) can be roughly interpreted as:
**<housing>-<platform>-<GPU model>-<VRAM per GPU>-<number of GPUs>**.

**MIG** stands for **Multi-Instance GPU**.
Developed by NVIDIA to be partition a GPU into smaller GPU instances, with its own dedicated resources.

## **GPUs**

| GPU | Brand | Arch. | Year | FP32 TFLOPS | Tensor Cores FP16 TFLOPS | Memory (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| P100 | NVDA | Pascal | 2016 | ~10 | - | 16 |
| V100 | NVDA | Volta | 2017 | ~16 | ~125 (1st Gen) | 16 |
| [A100](https://www.nvidia.com/en-us/data-center/a100/) | NVDA | Ampere | 2020 | ~20 | ~312 (3nd Gen) | 40/80 |
| [H100](https://www.nvidia.com/en-us/data-center/h100/) | NVDA | Hopper | 2022 | ~51 | ~1000 (4th Gen) | 94 |

## **Server**

|  | What it is | Build by | Usage |
| --- | --- | --- | --- |
| HGX | GPU baseboard & platform design | OEMs + NVIDIA | Used by partners to build servers |
| [DGX](https://www.nvidia.com/en-us/data-center/dgx-platform/) | Complete, ready-to-use AI system | NVIDIA | Turnkey AI/HPC server from NVIDIA |

[1. Access](/1-access-10746642.html)
[3. Storage](/3-storage-10746646.html)

[Impressum](https://www.lrz.de/impressum/) |
[Datenschutzerklärung](https://www.lrz.de/datenschutzerklaerungen/datenschutzerklaerung_confluence/)
|
[Barrierefreiheit](https://www.lrz.de/barrierefreiheit/)

null