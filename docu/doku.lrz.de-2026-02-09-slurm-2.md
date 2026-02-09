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

# 5.2 Slurm Batch Jobs - Single GPU

Batch jobs are **non-interactive** and the preferred way of using the LRZ AI Systems.
In a batch job, the allocation of resources and job submission are done in a single step.
If no resources are available, the job queues until the requested allocation is possible.

SLURM is a resource manager designed for multi-user systems. If the requested resources are not immediately available, the job is placed in a queue until the allocation can be granted. You can check the queue using squeue, or view only your own jobs with **squeue --me**. SLURM schedules jobs using policies such as **fair-share** to balance access among users. Batch jobs are executed automatically once scheduled.

**Relevant project documentations:** **[Enroot](https://github.com/NVIDIA/enroot)**, **[Slurm](https://slurm.schedmd.com/documentation.html)**, **[Pyxis](https://github.com/NVIDIA/pyxis)**

# **Slurm Essentials**

Jobs are typically executed through a combination of two key commands:

* **sbatch**
* **srun**
* **scancel (–me)**

**sbatch** is used to **submit a job script** to the queue. Slurm reads the #SBATCH directives inside the script to determine the requested resources (nodes, GPUs, time, etc.). Once submitted, the job waits in the queue until these resources become available. This step happens on the **login node** and defines the resource allocation for your job.

When the job starts, Slurm launches your script as a **batch shell** on the first allocated compute node. The script itself runs sequentially unless parallel execution is explicitly requested. Inside this script, each **srun** command creates a **job step,** which is a managed task or parallel workload that runs across one or more of your allocated nodes.

**scancel (–me)** allows you to cancel any job that you have submitted, whether it is running or pending.

A **Slurm job step** is a subdivision of a job and in practice, each **srun** command inside a job script creates a **job step**, allowing multiple parallel tasks (like: preprocessing, training, evaluation) to run sequentially or concurrently within the same job allocation.

This separation allows you to manage complex workflows where multiple steps run sequentially or in parallel within the same job allocation.

# **Examples**

## **Examples without Enroot**

The **`sbatch`** command submits batch scripts as jobs to the Slurm queue with:

```
sbatch job.sh
```

The following shows an example batch script, which is **not** explicitly using Enroot:

```
#!/bin/bash
#SBATCH -p lrz-v100x2                   # Select partition (use sinfo)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH -o log_%j.out                   # File to store standard output
#SBATCH -e log_%j.err                   # File to store standard error

echo "Start on $(hostname) at $(date)"  # Run outside of srun
srun command                            # Run the actual command GPU-enabled with srun
```

### Preamble

The first part of a batch script is the **preamble**, which includes lines starting with **#!** and **#SBATCH**.
This section defines the **resource allocation** required to run the job, such as the partition, number of GPUs, and runtime limits.

In addition, two important **#SBATCH** options specifywhere to redirect the job’s output and error messages**.** Since batch jobs are non-interactive, there is no terminal or shell to display output. Instead, the standard output and error streams must be written to files. In our example, we use log\_%j.out and log\_%j.err, where %j is automatically replaced by the **Slurm job ID**. A convenient way to dynamically monitor the output is:

```
tail -f log_<job_id>.*
```

### Commands

Following the preamble, the actual **job commands** are listed.
In this example, the script runs two commands sequentially.

The first command (echo) is **not run with srun**. It is executed directly by the SLURM batch script on the **first node** of the allocation.
Since it is outside of SLURM’s **job step** management, it does not benefit from features like resource binding or tracking.
This is fine for simple shell operations such as logging or environment setup.

The second command is run with **srun**, which means it is managed as a **SLURM job step**.
This launches the command in a parallel context, typically across **all nodes of the allocation** (unless specified otherwise).
If the allocation includes only a single node, **srun** will still create a parallel job, but limited to that one node.

Running with srun initializes **MPI-related environment variables** such as LOCAL\_RANK, RANK, and WORLD\_SIZE, which are often required by distributed frameworks (e.g., PyTorch, TensorFlow, MPI). These variables help coordinate parallel computation by assigning each process a role and identity within the job.

## **Examples with** **Enroot**

### **Single Container for Entire Job using Pyxis**

For all containerized jobs, even single-node workloads, we recommend using the **Pyxis** plugin by specifying the **--container-image** option.
Instead of using the Enroot command-line interface directly.

Although **srun** is not explicitly used before each command in the script, they are all executed as part of a parallel job.

Keep in mind that invoking **srun** within this context will fail, as it is not available inside the scope of an already parallel job.
Always specify absolute paths in the preamble. For #SBATCH directives, shell variables (e.g. $HOME) are not expanded during job submission.

```
#!/bin/bash
#SBATCH -p lrz-v100x2                   # Select partition (use sinfo)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH -o log_%j.out                   # File to store standard output
#SBATCH -e log_%j.err                   # File to store standard error
#SBATCH --container-images=/abs/path/to/nvidia+pytorch+23.10-py3 # (or 'nvcr.io/nvidia/pytorch:23.10-py3')
#SBATCH --container-mounts=/abs/path/to/project:/workspace       # Mount host project directory into /workspace

command1
command2
```

### **Separate Container per Job Step using Pyxis**

In Slurm, a **parallel job** is any job launched with **srun** (or **mpirun**). Each srun command creates a **job step** managed by Slurm. These job steps can run tasks across multiple nodes, set up environment variables for coordination (SLURM\_PROCID, RANK, WORLD\_SIZE), and ensure proper resource binding, tracking, and process control.

For containerized parallel jobs, even when allocating just a single node, we recommend using **[the Pyxis plugin](https://github.com/NVIDIA/pyxis)** for SLURM by specifying **--container-image** to manage container execution. With **--container-image**, you can either specify a container URL from the registry, which pulls the image each time you run the script (for example 'nvcr.io/nvidia/pytorch:23.10-py3') or reference a locally stored image for faster reuse.

Each job step, created with srun, is executed in a container created from the respective specified image, with containers instantiated on every allocated node.

**Note**: each call to **srun** results in the creation of a fresh container instance, even if the same image is used.

```
#!/bin/bash
#SBATCH -p lrz-v100x2                   # Select partition (use sinfo)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH -o log_%j.out                   # File to store standard output
#SBATCH -e log_%j.err                   # File to store standard error

srun --container-image='nvcr.io/nvidia/pytorch:23.10-py3' command1
srun --container-image='nvcr.io/nvidia/pytorch:23.11-py3' command2
srun --container-image='nvcr.io/nvidia/pytorch:23.12-py3' bash -c "command3 ; command4"
```

### **Using Enroot CLI (not recommended)**

To run **non-parallel containerized jobs** with SLURM using **Enroot**, you typically work with a pre-existing container image.
This involves two separate steps in your batch script:

1. Creating a container from the container image
2. Running the desired command inside the created container

The following script illustrates this approach:

```
#!/bin/bash
#SBATCH -p lrz-v100x2                   # Select partition (use sinfo)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH -o log_%j.out                   # File to store standard output
#SBATCH -e log_%j.err                   # File to store standard error

enroot create <NAME>.sqsh
enroot start NAME command
```

The option **--name** CNAME in **enroot create** would assign the container the name CNAME and create it on the first node of your allocation.
The line **`enroot start NAME command`** also executes the `command` in the first node of the allocation within the container.
[As of Ubuntu 22.04, using the `Enroot` command line interface for starting the job without previously creating the container is not possible](https://github.com/NVIDIA/enroot/issues/130#issuecomment-1196103352).

[5.1 Slurm Interactive Jobs](/5-1-slurm-interactive-jobs-1898974515.html)
[5.3 Slurm Batch Jobs - Multi GPU](/5-3-slurm-batch-jobs-multi-gpu-1898974517.html)

[Impressum](https://www.lrz.de/impressum/) |
[Datenschutzerklärung](https://www.lrz.de/datenschutzerklaerungen/datenschutzerklaerung_confluence/)
|
[Barrierefreiheit](https://www.lrz.de/barrierefreiheit/)

null