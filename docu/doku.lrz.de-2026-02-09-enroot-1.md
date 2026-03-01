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
* [4. Enroot](/4-enroot-10746639.html)

# 4.1 Enroot - Introduction

The LRZ AI Systems use the **[Enroot](https://github.com/NVIDIA/enroot/)** container framework provided by **NVIDIA**, to run jobs in isolated environments. Enroot allows users to define custom software stacks and operates entirely in user space.

**No root privileges** are needed to build or run containers. This approach ensures flexibility while maintaining system security and performance. Enroot supports various container image formats, including **Docker** images.

This means you can use any standard Docker image, such as those from **Docker Hub** or the **NVIDIA NGC** Cloud. Using NVIDIA NGC containers requires **[registration](https://ngc.nvidia.com/signin)**. For details, see section **[4.2 Enroot - Images from Nvidia NGC](/4-2-enroot-images-from-nvidia-ngc-1895502567.html)**.

**Please note**:

Enroot is **not** available on the SSH **login nodes** of the LRZ AI Systems. To use Enroot on the **compute nodes**, please start an interactive session, as described in **[5.1 Slurm Interactive Jobs](/5-1-slurm-interactive-jobs-1898974515.html)**.

Tools like module, conda, and pip are not officially supported on the LRZ AI Systems. The only supported method for installing custom software is through the use of **containers** with Enroot.

# **Enroot Runtime Configuration**

The runtime can be configured through the file **/etc/enroot/****`enroot.conf`** or by using environment variables in your **.bashrc**. Environment variables take precedence over the configuration file. On the AI Systems, the user’s home directory is mounted by default, and the container root filesystem is writable. You can disable this with:

```
echo 'export ENROOT_ROOTFS_WRITABLE=no' >> ~/.bashrc
echo 'export ENROOT_MOUNT_HOME=no' >> ~/.bashrc
```

# **The Enroot Workflow**

Running containerized applications with the Enroot framework follows a slightly different workflow compared to other container technologies. We recommend reviewing the Enroot documentation for a deeper understanding ( see **[here](https://github.com/NVIDIA/enroot/blob/master/doc/cmd/import.md)**). However, in most common use cases, you’ll only need three commands:

* **enroot import**
* **enroot create**
* **enroot start**

Other frequently used commands include **enroot export**, **enroot list**, and **enroot remove**.

## **Start an Interactive Session**

First, create an interactive allocation. The Enroot container runtime is **not** available on the SSH login nodes only on the compute nodes, therefore you need to start a Slurm interactive session using the following command. For more details, see **[5.1 Slurm Interactive Jobs](https://collab.dvb.bayern/spaces/LRZPUBLICDRAFT/pages/1897765154/5.1%2BSlurm%2BInteractive%2BJobs):**

```
salloc -p lrz-v100x2 --gres=gpu:1
srun --pty bash
```

## **Importing an Enroot Container Image**

To use a container image from a registry (e.g., Docker or NVIDIA NGC), start by importing it with the **enroot import** command. This converts the image into an Enroot-compatible .sqsh file, which can be reused to launch multiple containers.

```
enroot import docker://ubuntu
```

## **Creating an Enroot Container**

Once you have an Enroot image, use the **enroot create** to create the container. The command will expand the container image into a local container filesystem. Each container image requires a separate run of this command. To assign custom names to containers, see the **[official documentation](https://github.com/NVIDIA/enroot/blob/master/doc/usage.md)**. The following command will create the container, which will inherit the image name (without the .sqsh extension). You can assign a custom name to a container by using the **--name CUSTOM\_NAME** option after the **enroot create** command.

```
enroot create ubuntu.sqsh
```

## **Start an Enroot Container**

Once you have an Enroot container, you can run applications within its defined software environment using the enroot start command.
You can either execute a predefined hook script inside the container (refer to the container image documentation for its location)
or directly specify the executable as a command-line argument.

```
enroot start ubuntu
```

## **Install Software inside an Enroot Container**

If you need to run a command as root inside the container, for example to install software, you can use the **--root** option. Note that this grants root privileges only within the container environment, not on the host system.

```
enroot start --root --rw ubuntu
```

With root privileges inside the container you can now install your software stack. The software is installed on the writable layer of the container. For example, the following command installs essential tools to build Python packages like xformers or faiss from source.

```
apt update && apt install -y build-essential cmake && exit
```

## **Store an Enroot Container**

After exiting, you can list your containers with **enroot list**. To make the changes to container persistent, it needs to be exported into a container image (.sqsh file).

```
enroot export -o ubuntu-build.sqsh ubuntu
```

## **Container Cleanup**

Finally, you can list the available container on the compute node and remove not needed ones with:

```
enroot list
```

```
enroot remove ubuntu
```

**This is an educational example. To run your AI/ML workload efficiently use the highly optimized docker containers provided directly from Nvidia NGC.**

[4. Enroot](/4-enroot-10746639.html)
[4.2 Enroot - Images from Nvidia NGC](/4-2-enroot-images-from-nvidia-ngc-1895502567.html)

[Impressum](https://www.lrz.de/impressum/) |
[Datenschutzerklärung](https://www.lrz.de/datenschutzerklaerungen/datenschutzerklaerung_confluence/)
|
[Barrierefreiheit](https://www.lrz.de/barrierefreiheit/)

null