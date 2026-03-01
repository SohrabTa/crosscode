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

# 4.3 Enroot - GPU Enable Images

NVIDIA NGC offers a catalogue of containers covering a broad spectrum of software packages (see **[4.2 Enroot - Images from Nvidia NGC](/4-2-enroot-images-from-nvidia-ngc-1895502567.html)**). These containers supply the `CUDA Toolkit`, `cuDNN libraries`, and NVIDIA dependencies and **it recommended to use containers from NGC**. However, it is also possible to use containers from a different container registry or catalogue, which may not be optimized for Nvidia GPUs.

No matter where your container image comes from, your workload might depend on a package not provided by any image. This guide describes how to create a new custom Enroot container image by extending an existing one.

# **GPU Enable Container**

In general, the **CUDA driver** should always be installed on the **host**, while the **CUDA toolkit** can either be installed on the host or within the container. Therefore, there are two ways to enable GPU support inside a container:

1. **Installing the NVIDIA Container Toolkit within the container:** This approach is documented by NVIDIA in their official guide **[here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)**.
2. **Passing the CUDA toolkit and driver utilities from the host to the container:**For simplicity, we focus on this pass-through method, which only requires setting a few environment variables inside the container.

The following environment variables control how CUDA is exposed to the container (see [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html) for more information):

* **NVIDIA\_DRIVER\_CAPABILITIES** – specifies which driver features the container should access (e.g., compute, utility, video, or graphics).
* **NVIDIA\_VISIBLE\_DEVICES** – determines which GPUs are visible to the container.

After identifying the variables and their values according to your needs, add them to **/etc/environment** inside your container.

The following example shows a typical configuration:

```

```

```
echo "NVIDIA_DRIVER_CAPABILITIES=compute,utility" >> /etc/environment
echo "NVIDIA_VISIBLE_DEVICES=all" >> /etc/environment
```

## **Avoiding Conflicts from Preinstalled Libraries**

The presence of Nvidia libraries within a container image might produce crashes. If you are not using a container from NGC, be sure your image does not include:

* The CUDA toolkit library
* The Nvidia libcontainer toolkit library (libnvidia-container)

Let the Enroot runtime add these required libraries to your containers.

# **Custom Image Example**

This example is for teaching purposes only and does not represent a real use case. The described functionality is already provided by NGC containers.

## **Start an Interactive Session**

First, create an interactive allocation. Since Enroot is only available on the compute nodes, you need to start a Slurm interactive session using the following command. For more details, see **[5.1 Slurm Interactive Jobs](/5-1-slurm-interactive-jobs-1898974515.html):**

```
salloc -p lrz-v100x2 --gres=gpu:1
srun --pty bash
```

## **Import a Base Image**

Import the Ubuntu base image. This command will return an error if ubuntu.sqsh already exists.

```
enroot import docker://ubuntu
enroot create ubuntu.sqsh
```

## **GPU Enable the Container**

If you need to modify the container contents during runtime (e.g., install software, change configurations), you must start the container with the **--rw** option to enable write access. Additionally, if these modifications require administrative privileges, use the **--root** option to run with root permissions inside the container. However, on the AI Systems the **--rw** option is set per default in the config file: enroot.conf.

Now, pass the CUDA driver and CUDA toolkit from the host to the container by setting the following environment variables when starting it:

```
enroot start --root --rw --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env NVIDIA_VISIBLE_DEVICES=all ubuntu
```

To make the environment persistent in the final exported Enroot image, add the variables to the **/etc/environment** file:

```
echo "NVIDIA_DRIVER_CAPABILITIES=compute,utility" >> /etc/environment
echo "NVIDIA_VISIBLE_DEVICES=all" >> /etc/environment
```

## **Install Software**

In the following steps, we will install packages in the container using the distribution’s package manager, create a Python virtual environment, and add a small snippet to the container’s **.bashrc** file to automatically activate the virtual environment by default.

Install packages using the **apt** package manager:

```
apt-get update && apt-get install -y build-essential cmake && exit
```

## **Persist the Changes**

Finally, exit the container and export it.

```
enroot export -o ubuntu-cuda.sqsh ubuntu && enroot create ubuntu-cuda.sqsh
```

[4.2 Enroot - Images from Nvidia NGC](/4-2-enroot-images-from-nvidia-ngc-1895502567.html)
[5. Slurm](/5-slurm-1897076524.html)

[Impressum](https://www.lrz.de/impressum/) |
[Datenschutzerklärung](https://www.lrz.de/datenschutzerklaerungen/datenschutzerklaerung_confluence/)
|
[Barrierefreiheit](https://www.lrz.de/barrierefreiheit/)

null