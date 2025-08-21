# Jupyter Enterprise Gateway: Powering Remote Kernel Access for Jupyter Notebooks

**Jupyter Enterprise Gateway** ([GitHub Repository](https://github.com/jupyter-server/enterprise_gateway)) enables secure and scalable access to Jupyter kernels in a distributed computing environment, allowing users to leverage powerful remote resources for their data science and analysis workflows.

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

**Key Features:**

*   **Remote Kernel Launch:** Launch kernels on remote clusters, including Apache Spark managed by YARN, IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Supported Kernels:** Out-of-the-box support for Python (IPython kernel), R (IRkernel), and Scala (Apache Toree kernel).
*   **Flexible Deployment:** Supports kernel launching on local servers, specific cluster nodes, and nodes identified by resource managers.
*   **Secure Communication:** Ensures secure communication between clients, the Enterprise Gateway server, and kernels.
*   **Multi-Tenant Capabilities:** Enables multiple users to access and utilize the Gateway.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for continuous operation.
*   **Customizable Profiles:** Allows users to associate configuration settings with specific kernels.
*   **Integration with JupyterHub:** For managing multiple Jupyter Notebook deployments, consider using [JupyterHub](https://github.com/jupyterhub/jupyterhub).

## Technical Overview

Jupyter Enterprise Gateway expands upon the functionality of Jupyter Kernel Gateway by providing robust features for accessing remote kernels within an enterprise environment. Enterprise Gateway provides a [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html) and adds additional capabilities.

**Key Technical Aspects:**

*   **Remote Kernel Hosting:** Enables launching kernels across a distributed cluster.
*   **Resource Manager Support:** Integrates with resource managers like YARN, Kubernetes and Docker Swarm.
*   **Extensible Framework:** Allows for the integration of additional resource managers and kernel types.

  ![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Get started with Jupyter Enterprise Gateway using `pip`:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

For detailed installation and configuration information, please refer to the comprehensive documentation:

*   [User Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html)
*   [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway)

## System Architecture

Learn more about the architecture and internal components of Jupyter Enterprise Gateway:

*   [System Architecture](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html)

## Contributing

Contribute to the development of Jupyter Enterprise Gateway and help improve this open-source project:

*   [Contribution Guidelines](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html)
*   [Development Environment Setup](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html)