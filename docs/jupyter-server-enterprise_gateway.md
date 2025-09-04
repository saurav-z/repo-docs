# Jupyter Enterprise Gateway: Seamlessly Launch Remote Kernels in Your Enterprise

**Jupyter Enterprise Gateway** ([GitHub Repository](https://github.com/jupyter-server/enterprise_gateway)) empowers Jupyter Notebook users to access and leverage computational resources in a distributed cluster environment. This open-source project allows Jupyter Notebooks to launch remote kernels, enhancing data science and computational workflows.

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

**Key Features:**

*   **Remote Kernel Launch:** Enables Jupyter Notebook to launch kernels on remote resources within your enterprise infrastructure.
*   **Cluster Support:** Out-of-the-box integration with popular cluster managers, including Apache Spark (managed by YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Supported Kernels:** Ready-to-use kernels for Python (using IPython kernel), R (using IRkernel), and Scala (using Apache Toree kernel).
*   **Flexible Kernel Deployment:** Supports launching kernels locally or on specific cluster nodes, including resource manager-identified nodes.
*   **Secure Communication:** Provides secure communication channels from the client, through the Enterprise Gateway server, to the kernels.
*   **Multi-Tenant Capabilities:** Designed with multi-tenancy in mind, enabling efficient resource sharing.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for uninterrupted workflows.
*   **Configurable Profiles:** Allows users to associate profiles with custom configuration settings to a kernel.

**For detailed information, please see the following resources:**

*   **[Website](https://jupyter-enterprise-gateway.readthedocs.io/)**
*   **[Technical Overview](#technical-overview)**
*   **[Installation](#installation)**
*   **[System Architecture](#system-architecture)**
*   **[Contributing](#contributing)**
*   **Full Documentation:** [https://jupyter-enterprise-gateway.readthedocs.io/en/latest](https://jupyter-enterprise-gateway.readthedocs.io/en/latest)

Jupyter Enterprise Gateway mirrors the functionality of Kernel Gateway's [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html).  If you are looking to manage multiple Jupyter Notebook deployments, please consider using [JupyterHub](https://github.com/jupyterhub/jupyterhub).

## Technical Overview

Jupyter Enterprise Gateway is a web server providing headless access to Jupyter kernels within an enterprise environment. It extends the functionality of Jupyter Kernel Gateway to support remote kernel deployments, offering features beyond Kernel Gateway's standard [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html):

*   **Remote Kernel Hosting:** Provides support for remote kernels hosted throughout the enterprise, supporting local and cluster-based kernel launching.
*   **Resource Manager Integration:** Supports various resource managers like YARN, IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Extensible Framework:** Designed with an extensible framework to accommodate other resource managers.
*   **Secure Communication:** Uses secure communication for data transfer between the client and the kernel.
*   **Multi-Tenant Capabilities:** Built with multi-tenancy in mind, for efficient resource usage.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for continuous operation.
*   **Profile Association:** Support for associating profiles consisting of configuration settings to a kernel for a given user (see [Project Roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html)).

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Install Jupyter Enterprise Gateway using `pip`:

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

Consult the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) for detailed installation instructions. Review the [configuration options within the Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) for information about the supported options.

## System Architecture

The [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html) provides details about the architecture of Enterprise Gateway, including its remote kernel, process proxy, and launcher frameworks.

## Contributing

The [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html) outlines how to contribute to Enterprise Gateway, including the project roadmap, and setup of a development environment [here](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html).