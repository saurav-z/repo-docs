# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels for Your Enterprise

**[Visit the official website](https://jupyter-enterprise-gateway.readthedocs.io/)** |
**[Technical Overview](#technical-overview)** |
**[Installation](#installation)** |
**[System Architecture](#system-architecture)** |
**[Contributing](#contributing)**

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

Jupyter Enterprise Gateway empowers your enterprise to leverage the full potential of Jupyter Notebooks by enabling secure, remote kernel execution on distributed clusters. This allows you to scale your data science and analytical workloads across a wide range of environments.

**Key Features:**

*   **Remote Kernel Execution:** Launch kernels on remote clusters, including Apache Spark managed by YARN, IBM Spectrum Conductor, Kubernetes, or Docker Swarm.
*   **Broad Kernel Support:** Out-of-the-box support for popular kernels:
    *   Python (via IPython kernel)
    *   R (via IRkernel)
    *   Scala (via Apache Toree kernel)
*   **Secure Communication:** Secure communication between the client, Enterprise Gateway server, and kernels.
*   **Multi-Tenancy:** Supports multi-tenant environments, allowing multiple users and teams to share resources.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for uninterrupted workflows.
*   **Extensible Framework:** Extensible framework enables configuration for other environments beyond the out-of-the-box support.
*   **Resource Manager Integration:** Integrated support for popular resource managers (YARN, Kubernetes, etc.).

**For detailed documentation, please visit the [Jupyter Enterprise Gateway documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest).**

**Please Note:** Jupyter Enterprise Gateway is designed for managing remote kernels; for managing multiple Jupyter Notebook deployments, please consider [JupyterHub](https://github.com/jupyterhub/jupyterhub).

## Technical Overview

Jupyter Enterprise Gateway builds upon Jupyter Kernel Gateway, offering enhanced capabilities for enterprise environments. It provides feature parity with Kernel Gateway's [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html) in addition to the following:

*   **Flexible Kernel Deployment:** Launch kernels:
    *   Locally
    *   On specific nodes in the cluster using a round-robin algorithm
    *   On nodes identified by an associated resource manager
*   **Profiles:** Ability to associate profiles consisting of configuration settings to a kernel for a given user (see [Project Roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html))

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Get started quickly with these commands:

```bash
# Install from PyPI
pip install --upgrade jupyter_enterprise_gateway

# View all configuration options
jupyter enterprisegateway --help-all

# Run with default options
jupyter enterprisegateway
```

For comprehensive configuration options, consult the [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

## System Architecture

The [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html) details the inner workings of Enterprise Gateway, including its remote kernel, process proxy, and launcher frameworks.

## Contributing

We welcome contributions!  See the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html) for information on how to contribute.  You'll also find information on [setting up a development environment](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html) and typical developer tasks.

**[Back to the original repository](https://github.com/jupyter-server/enterprise_gateway)**