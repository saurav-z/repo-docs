# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels in Your Enterprise

**[Website](https://jupyter-enterprise-gateway.readthedocs.io/)** |
**[Technical Overview](#technical-overview)** |
**[Installation](#installation)** |
**[System Architecture](#system-architecture)** |
**[Contributing](#contributing)**

Jupyter Enterprise Gateway (JEG) empowers data scientists and researchers to leverage the scalability and resources of distributed clusters directly from their Jupyter Notebooks. Find the original repository [here](https://github.com/jupyter-server/enterprise_gateway).

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

Key Features of Jupyter Enterprise Gateway:

*   **Remote Kernel Support:** Launch Jupyter kernels on remote clusters, including Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Out-of-the-Box Kernel Support:** Ready to use with Python (IPython), R (IRkernel), and Scala (Apache Toree).
*   **Secure Communication:** Ensures secure communication between the client, Enterprise Gateway, and the kernels.
*   **Multi-Tenancy:** Provides multi-tenant capabilities for shared environments.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for continuous work.
*   **Extensible Framework:** Allows for the configuration of other resource managers beyond those listed.

For more details, consult the full documentation: [https://jupyter-enterprise-gateway.readthedocs.io/en/latest](https://jupyter-enterprise-gateway.readthedocs.io/en/latest)

*Note: Jupyter Enterprise Gateway focuses on remote kernel management; for managing multiple Jupyter Notebook deployments, consider [JupyterHub](https://github.com/jupyterhub/jupyterhub).*

## Technical Overview

Jupyter Enterprise Gateway extends the functionality of Jupyter Kernel Gateway, adding key features for enterprise environments. It supports remote kernels, offering flexibility in kernel launching: locally, on specific cluster nodes, or via resource managers.  This includes:

*   Support for launching kernels locally or on specific cluster nodes.
*   Support for Apache Spark managed by YARN, IBM Spectrum Conductor, Kubernetes or Docker Swarm.
*   Support for associating profiles with kernels for configuration settings (see [Project Roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html)).

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Install Jupyter Enterprise Gateway easily using `pip`:

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

Detailed instructions are in the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) and configuration options are in the [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

## System Architecture

The [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html) offers insights into the internal workings of Enterprise Gateway, including its remote kernel, process proxy, and launcher frameworks.

## Contributing

Learn how to contribute to Jupyter Enterprise Gateway, including setting up a development environment and understanding typical developer tasks, by visiting the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html).