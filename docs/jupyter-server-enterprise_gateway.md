# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels in Your Enterprise

**[Website](https://jupyter-enterprise-gateway.readthedocs.io/)** |
**[Technical Overview](#technical-overview)** |
**[Installation](#installation)** |
**[System Architecture](#system-architecture)** |
**[Contributing](#contributing)**

Jupyter Enterprise Gateway (Jupyter EG) empowers your Jupyter Notebook environment by enabling the seamless launch and management of remote kernels across distributed clusters, including Kubernetes, Apache Spark, and more. **This allows data scientists and engineers to leverage the resources of their enterprise environment directly from their Jupyter notebooks.**

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

**Key Features:**

*   **Remote Kernel Support:** Launch kernels on remote clusters such as Kubernetes, Apache Spark (YARN), IBM Spectrum Conductor, and Docker Swarm.
*   **Enhanced Security:** Secure communication between the client, gateway, and kernels.
*   **Multi-Tenant Capabilities:** Supports multiple users and environments.
*   **Persistent Kernel Sessions:** Maintain kernel sessions for uninterrupted work.
*   **Extensible Framework:** Easily configure support for additional resource managers.
*   **Pre-built Kernel Support:** Out-of-the-box support for Python (with IPython), R (with IRkernel), and Scala (with Apache Toree).
*   **Flexible Deployment:** Launch kernels locally or on specific nodes using round-robin or resource manager-driven allocation.

For a comprehensive overview of the available kernels, consult the [documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest). Note that Jupyter Enterprise Gateway is designed for managing remote kernels; for managing multiple Jupyter Notebook deployments, consider using [JupyterHub](https://github.com/jupyterhub/jupyterhub).

## Technical Overview

Jupyter Enterprise Gateway is a web server that provides headless access to Jupyter kernels. It expands on the functionality of Jupyter Kernel Gateway.

**Key Features of Jupyter EG (in addition to Kernel Gateway's WebSocket mode):**

*   **Remote Kernel Launching:** Supports launching kernels:
    *   Locally on the Enterprise Gateway server (like Kernel Gateway).
    *   On specific cluster nodes via a round-robin approach.
    *   On nodes managed by an associated resource manager.
*   **Cluster Support:**  Native support for Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm. The extensible framework allows for integration with other resource managers.
*   **Secure Communication:** Encrypted data flow for secure operation.
*   **Multi-Tenancy:**  Supports multiple users and their individual kernel environments.
*   **Persistent Kernels:**  Kernel sessions are retained for continuous operation.
*   **Profile Management:**  Associate kernel configuration settings with profiles for users (see the [Project Roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html) for details).

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Get started with Jupyter Enterprise Gateway quickly using `pip`:

```bash
# Install from PyPI
pip install --upgrade jupyter_enterprise_gateway

# Show all configuration options
jupyter enterprisegateway --help-all

# Run with default options
jupyter enterprisegateway
```

Detailed installation instructions and configuration options are available in the [User Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) and the [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway), respectively.

## System Architecture

The [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html) provides in-depth information about the remote kernel, process proxy, and launcher frameworks.

## Contributing

We welcome contributions! Learn how to contribute, set up a development environment, and find developer tasks on the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html), including the [developer installation guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html).

**[Back to the GitHub Repository](https://github.com/jupyter-server/enterprise_gateway)**