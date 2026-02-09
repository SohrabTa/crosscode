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

# 3. Storage

The LRZ AI Systems are integrating the full offer of the **[LRZ Data Science Storage (DSS)](/data-science-storage-10745685.html)** system.

# **Storage Options**

| Storage Pool | Use Case | Top-level Directory | Size Limit | Automated Backup | Expiration |
| --- | --- | --- | --- | --- | --- |
| Home Directory | Critical files like code, scripts, and configs that need regular backups and are small. | /dss/dsshome1/ | 100 GB | Yes, backup to tape and file system snapshots | Lifetime of LRZ project |
| **AI Systems DSS** | **High-bandwidth, low-latency storage for I/O.**  use this for reading and writing model data and results | /dss/dssfs04 | 4 TB (up to) 4+ TB (costs apply) | No for free tier / Yes for paid option | Until further notice |
| Linux Cluster DSS | General purpose, long-term storage. | /dss/dssfs02 /dss/dssfs03 /dss/dssfs05 | 10 TB (up to) 20+ TB (costs apply) | No for free tier / Yes for paid option | Lifetime of the data project |
| Private DSS | System Owner Defined | /dss/dsslegfs01 /dss/dsslegfs02 /dss/dssmcmlfs01 /dss/mcmlscratch | System Owner Defined | System Owner Defined | System Owner Defined |

## **Home Directory**

Home directories can be accessed via SSH on the **login nodes** (**[login.ai.lrz.de](http://login.ai.lrz.de)**) or through the **web frontend** at **[https://login.ai.lrz.de](https://login.ai.lrz.de/)**. When logging in via terminal, you are placed directly in your **home directory**. By typing `pwd` immediately after login, you can verify this and see the full path to your current (home) directory.

The LRZ AI Systems share a unified home directory with the LRZ Linux Cluster (see **[File Systems and IO on Linux-Cluster](/file-systems-and-io-on-linux-cluster-10745972.html)**). Home directories are hosted in a dedicated DSS container managed by LRZ. They are limited in both capacity and I/O performance (bandwidth and latency) and are therefore **NOT suitable** for high-intensity AI workloads or large-scale data operations.

The home directory should primarily be used to store **code, configuration files,** and **other lightweight data**. All home directories are regularly backed up to ensure data integrity and security.

## **AI Systems DSS**

A dedicated **AI Systems DSS** provides high-performance, SSD-based network storage designed for demanding AI workloads. It is optimized for **high-bandwidth, low-latency I/O operations** to support the data-intensive requirements of modern AI applications. In contrast to the home directory, the **AI Systems DSS is the appropriate location** for high-intensity AI workloads and large-scale data operations.

Access to the AI DSS storage is granted upon request by the **Master User** of an LRZ project via the LRZ Servicedesk using the following **[form](https://servicedesk.lrz.de/en/ql/createsr/21)**. A quota of up to 4 TB, 2 million files, and a maximum of 3 DSS containers (typically a single container) can be allocated.
Once provisioned, the Master User assumes the role of **DSS Data Curator** and is responsible for managing the assigned storage quotas within their project.

Additional AI Systems DSS storage (4 TB or more) can be requested as part of the **[DSS on demand offer](/data-science-storage-10745685.html)**, which is subject to additional costs. To request a higher file number limit, please submit a ticket via our [Servicedesk](http://servicedesk.lrz.de) and select AI Systems under High Performance Computing.

## **Linux Cluster DSS**

The Master User can request additional storage of up to 10 TB. To do so, they must first request the activation of the project for the Linux Cluster via the LRZ Servicedesk using the following **[form](https://servicedesk.lrz.de/en/ql/createsr/12)**, and then submit a separate request for explicit Linux Cluster DSS storage via this **[form](https://servicedesk.lrz.de/ql/createsr/23)**.

The **Linux Cluster DSS** is primarily intended for **long-term data storage** and general-purpose workloads. Both the home directory and the Linux Cluster DSS are **not designed for** high-intensity AI workloads or large-scale data operations.

## **Private DSS**

As part of a joint project offering, dedicated DSS systems can be purchased, deployed, and operated exclusively for a private group of users. For more information see **[here](https://www.lrz.de/en/service-center/alle-lrz-services/speicherloesungen/data-science-storage)** and **[here](/data-science-storage-10745685.html)**.

# **The GPFS Distributed File System**

The AI Systems at LRZ use the IBM General Parallel File System (GPFS), as their main storage backend. GPFS is a **high-performance distributed file system** designed for large-scale HPC environments. Unlike local file systems, which operate on a single disk or node, GPFS spreads data and metadata across many servers and disks, enabling parallel access by thousands of compute nodes.

## **Latency and I/O Patterns:**

Tasks that are instant on local disks, like creating, moving, or deleting many small files, can be slow on GPFS because metadata and file data are managed across multiple servers. For best performance, use **fewer** but **large files** and **sequential access** instead of many small random reads or writes.

## **Metadata and Inodes**

Each file or directory is represented by an **inode** that stores ownership, permissions, and disk location. In distributed systems, inode management adds overhead. Large numbers of files can quickly cause performance drops or reach inode limits.

## **Machine Learning Datasets**

Machine learning datasets often consist of millions of small files (e.g., individual images). This pattern leads to heavy metadata (inodes) traffic and slow data access. To improve efficiency:

* Pack data into **archive formats** (e.g., .tar, .zip, HDF5, or TFRecord).
* Avoid frequent directory scans (ls, find, stat) on large folders.

# **Additional Information**

Additional information can be found at **[File Systems and IO on Linux-Cluster](/file-systems-and-io-on-linux-cluster-10745972.html)** and **[Data Science Storage](/data-science-storage-10745685.html)**.

Use the following command on the login nodes to get an storage utilization overview of all individually accessible DSS containers:

```
dssusrinfo all
```

[2. Compute](/2-compute-10746641.html)
[4. Enroot](/4-enroot-10746639.html)

[Impressum](https://www.lrz.de/impressum/) |
[Datenschutzerklärung](https://www.lrz.de/datenschutzerklaerungen/datenschutzerklaerung_confluence/)
|
[Barrierefreiheit](https://www.lrz.de/barrierefreiheit/)

null