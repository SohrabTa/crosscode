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

# 4.2 Enroot - Images from Nvidia NGC

The Enroot container framework can import any Docker images and spin up containers from them. However, for LRZ AI Systems we strongly recommend using NVIDIA’s container images from **[the NVIDIA NGC catalog](https://ngc.nvidia.com/catalog)**. **NGC images are guaranteed to be fully optimized for NVIDIA GPUs in multi-GPU or multi-node environments** with regular maintenance and updates directly from NVIDIA.

# **The NVIDIA NGC Catalog**

Usage of the NGC catalog requires online registration and API key authentication from the machine downloading the container.

## **Register**

Register by creating an NVIDIA profile here: **<https://ngc.nvidia.com/signin>**.

Once logged in, you can already search for container images by keyword.
Go to the “Containers” tab, enter a keyword (e.g., *Pytorch*) in the search box, and press Enter.

Clicking on a container shows details such as the image name and tag.
While the format on NGC follows Docker conventions,
the key information is the image name, which you’ll need for the import with Enroot.

## **Authenticate**

Authenticate yourself from the machine to download images to the AI-Systems.
The following is a step-by-step guide to generating an **NVIDIA NGC API key** and configuring it for use with Enroot:

* Sign in to the NVIDIA NGC website: **<https://ngc.nvidia.com/catalog>**
* Click your user icon (top right) and select **Setup**
* Click **Get API Key**, **Generate API Key** and **save** it
* Login to the AI System and create a file named **~/enroot/.credentials** with following content:

```
machine nvcr.io login $oauthtoken password <KEY>
machine authn.nvidia.com login $oauthtoken password <KEY>

```

**Be sure that that you include a new line at the end.** After doing this, you are ready to start importing NVIDIA NGC containers.

## **Image Import**

For example, the following command imports the PyTorch container image (tag: 23.10-py3)

```

```

```
enroot import docker://nvcr.io/nvidia/pytorch:23.10-py3
```

Make sure to select the correct container version and verify that the preinstalled packages meet your needs. Ensure compatibility between the container and the system setup. For example, by checking the installed **NVIDIA driver** version using **nvidia-smi** on the compute node. It’s also important to confirm the CUDA version and other dependencies align with what’s expected by the container.

[4.1 Enroot - Introduction](/4-1-enroot-introduction-1895502566.html)
[4.3 Enroot - GPU Enable Images](/4-3-enroot-gpu-enable-images-2111055144.html)

[Impressum](https://www.lrz.de/impressum/) |
[Datenschutzerklärung](https://www.lrz.de/datenschutzerklaerungen/datenschutzerklaerung_confluence/)
|
[Barrierefreiheit](https://www.lrz.de/barrierefreiheit/)

null