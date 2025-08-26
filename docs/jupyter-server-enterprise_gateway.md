# Jupyter Enterprise Gateway: Unlock Distributed Computing for Jupyter Notebooks

**Jupyter Enterprise Gateway** empowers Jupyter Notebook users to harness the power of remote kernels in distributed computing environments like Kubernetes and Apache Spark, enabling scalable data analysis and model building. ([See the original repository](https://github.com/jupyter-server/enterprise_gateway))

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

**Key Features:**

*   **Remote Kernel Support:** Launch kernels on various compute clusters, including YARN, IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Out-of-the-Box Kernel Support:** Ready-to-use kernels for Python (IPython), R (IRkernel), and Scala (Apache Toree).
*   **Flexible Kernel Launching:** Launch kernels locally or on specific cluster nodes, including those managed by resource managers.
*   **Secure Communication:** Secure communication channels between the client, the Enterprise Gateway server, and the kernels.
*   **Multi-Tenant Capabilities:** Supports multiple users and projects within a single deployment.
*   **Persistent Kernel Sessions:** Maintain kernel sessions for uninterrupted workflows.
*   **Extensible Framework:** Easily configure and extend support for additional cluster environments.

## Technical Overview

Jupyter Enterprise Gateway acts as a web server providing headless access to Jupyter kernels.  It offers feature parity with Jupyter Kernel Gateway's [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html), plus advanced capabilities:

*   Supports launching kernels locally or remotely on cluster nodes using a round-robin algorithm or through a resource manager.
*   Provides built-in support for Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   Offers multi-tenancy and persistent kernel sessions.
*   Enables the association of configuration profiles to kernels for specific users (see [Project Roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html)).

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Get started with Jupyter Enterprise Gateway using `pip`:

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

Detailed installation instructions are available in the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html). Explore the [configuration options within the Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

## System Architecture

The [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html) provides insights into Enterprise Gateway's remote kernel, process proxy, and launcher frameworks.

## Contributing

Contribute to Jupyter Enterprise Gateway by following the guidelines outlined on the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html), including details on setting up a development environment ([devinstall](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html)) and common developer tasks.