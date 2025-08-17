# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels

**[Website](https://jupyter-enterprise-gateway.readthedocs.io/)** |
**[Documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest)** |
**[Installation](#installation)** |
**[System Architecture](#system-architecture)** |
**[Contributing](#contributing)**

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

Jupyter Enterprise Gateway (JEG) empowers data scientists and researchers by enabling Jupyter Notebooks to seamlessly connect to and utilize remote kernels in distributed computing environments.

**Key Features:**

*   **Remote Kernel Support:** Launch kernels on remote clusters including Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Extensible Framework:** Easily configure JEG to support other resource managers and kernel types.
*   **Secure Communication:** Ensures secure communication between the client, the gateway server, and the kernels.
*   **Multi-Tenant Capabilities:** Enables multiple users to share and utilize the gateway.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for uninterrupted workflows.
*   **Pre-built Kernel Support:** Out-of-the-box support for Python (IPython kernel), R (IRkernel), and Scala (Apache Toree kernel).
*   **Profile Management:** Associate profiles with kernel configurations for streamlined, personalized kernel experiences (see [Project Roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html)).

For managing multiple Jupyter Notebook deployments, consider using [JupyterHub](https://github.com/jupyterhub/jupyterhub).

## Technical Overview

Jupyter Enterprise Gateway extends Jupyter Kernel Gateway, providing headless access to Jupyter kernels in an enterprise environment, building on the functionality of [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html). It supports remote kernels launched:

*   Locally (like Kernel Gateway)
*   On specific cluster nodes (round-robin)
*   On nodes managed by resource managers

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Detailed installation instructions are available in the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html).

**Quick Start with `pip`:**

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

See the [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) for supported configuration options.

## System Architecture

Learn more about Enterprise Gateway's remote kernel, process proxy, and launcher frameworks on the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

## Contributing

Contribute to Jupyter Enterprise Gateway!  Find information on the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html), including the project roadmap, setting up a development environment, and developer tasks.

**[Back to the project on GitHub](https://github.com/jupyter-server/enterprise_gateway)**