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

# 5. Slurm

[Slurm](https://slurm.schedmd.com) is the open-source **resource manager** and **job scheduler** for Linux clusters of any size.
It allocates computing resources, runs and monitors jobs, and manages job queues, with optional plugins for advanced scheduling and accounting.

## **S-Commands**

The following are the most frequently used **Slurm commands** that users rely on for day-to-day work. They cover the essential steps of checking resources, submitting jobs, monitoring progress, and managing running or completed jobs.

* **sinfo** - Show available partitions, nodes, and their status.
* **squeue** - Display currently queued and running jobs.
* **srun** - Submit or launch a job (interactive or batch).
* **sbatch** - Submit a batch job script to the scheduler.
* **scancel** - Cancel a running or pending job.
* **salloc** - Allocate resources for an interactive job session.
* **sacct** - View job accounting and usage information.

## **Key Points**

The following key points describe important details specific to the **Slurm setup on the AI Systems**.
Make sure to follow these conventions when submitting or running jobs to ensure your workloads start and run correctly.

* Always specify the number of GPUs when requesting resources using: **--gres=gpu:1** (more general --gres=gpu:<number\_of\_GPUs>).
* For individual jobs and allocations, the **default time limit** is **1 hour**, and the **maximum time limit** is **2 days**.

## **Overview**

* [5.1 Slurm Interactive Jobs](/5-1-slurm-interactive-jobs-1898974515.html)
* [5.2 Slurm Batch Jobs - Single GPU](/5-2-slurm-batch-jobs-single-gpu-1898974516.html)
* [5.3 Slurm Batch Jobs - Multi GPU](/5-3-slurm-batch-jobs-multi-gpu-1898974517.html)
* [5.4 Slurm Batch Jobs - Multi Node](/5-4-slurm-batch-jobs-multi-node-1898714389.html)

[4.3 Enroot - GPU Enable Images](/4-3-enroot-gpu-enable-images-2111055144.html)
[5.1 Slurm Interactive Jobs](/5-1-slurm-interactive-jobs-1898974515.html)

[Impressum](https://www.lrz.de/impressum/) |
[Datenschutzerklärung](https://www.lrz.de/datenschutzerklaerungen/datenschutzerklaerung_confluence/)
|
[Barrierefreiheit](https://www.lrz.de/barrierefreiheit/)

null