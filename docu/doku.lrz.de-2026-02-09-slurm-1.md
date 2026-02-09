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
* [5. Slurm](/5-slurm-1897076524.html)

# 5.1 Slurm Interactive Jobs

Interactive jobs allow you to work directly within an allocated set of resources.
To view available resources such as partitions and the current status of compute nodes use the ***sinfo*** command.

## **Allocate Resources**

Allocate resources interactively with the **salloc** command. For example, to request one GPU in the **lrz-v100x2** partition, run:

```
salloc -p lrz-v100x2 --gres=gpu:1
```

The ***--gres=gpu:1*** option explicitly requests one GPU and is **essential** when working with GPU-enabled partitions.
Without this option, GPU resources will not be allocated.

You can adjust the number to request multiple GPUs, e.g., ***--gres=gpu:2*** for two GPUs.
This option is mandatory for all GPU partitions, with the sole exception of the lrz-cpu partition, which does not provide GPUs.

## **Launch Interactive Jobs**

Interactive jobs can be launched within an existing allocation using the **srun** command.
For example, the following command opens an interactive bash shell directly on the allocated compute node:

```
srun --pty bash
```

This gives you interactive access to the compute node, including all allocated resources.
You can now run commands directly and also use tools like **Enroot**, which are available within the compute node environment.

At this point, starting an Enroot container using the Enroot CLI **(enroot import**, **enroot create**, and **enroot start**as described in *Section***[4.1 Enroot - Introduction](/4-1-enroot-introduction-1895502566.html)**)is a natural next step. Setup your software environment within the interactive session.

## **Launch Interactive Container Jobs**

You can also run interactive jobs directly inside an **Enroot container** using the **[Pyxis plugin](https://github.com/NVIDIA/pyxis).**

To do this, first choose a suitable container image. It can come from **NGC**, **Docker Hub**, or be a already downloaded **local image**.Make sure the image includes all the required software and libraries (see: **[4.3 Enroot - Custom Images](/4-3-enroot-custom-images-1895502568.html)**).

Now pass the image to srun using the **--container-image** option.
SLURM will automatically create and launch a container from that image and run the specified command inside it.

The following command starts an interactive bash shell in a NGC PyTorch container.
Keep in mind, that using the repository URL will downloaded the image every time, which is slow (**--container-image='[nvcr.io/nvidia/pytorch:23.10-py3](http://nvcr.io/nvidia/pytorch%3A23.10-py3)'**)**.**
For frequently used images, download them once and reference the local image path instead.

```
srun --pty --container-mounts=$HOME/ai-systems-examples/:/workspace --container-image=$HOME/nvidia+pytorch+23.10-py3.sqsh bash
```

To make your project folder available inside the container, the **--container-mounts** option mounts that host folder into the container(Host:Container).

Additionally, the ***-*-container-name** option assigns a name to the container so it can be reused within the same job allocation.
Unnamed containers are deleted after the job, while named ones persist. However, it is not designed to persist across separate job allocations.
For details on this limitation and the reasoning behind it, see this **[discussion](https://github.com/NVIDIA/pyxis/issues/30#issuecomment-717654607)**.

[5. Slurm](/5-slurm-1897076524.html)
[5.2 Slurm Batch Jobs - Single GPU](/5-2-slurm-batch-jobs-single-gpu-1898974516.html)

[Impressum](https://www.lrz.de/impressum/) |
[Datenschutzerklärung](https://www.lrz.de/datenschutzerklaerungen/datenschutzerklaerung_confluence/)
|
[Barrierefreiheit](https://www.lrz.de/barrierefreiheit/)

null