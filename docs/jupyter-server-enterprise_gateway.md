# Jupyter Enterprise Gateway: Powering Remote Kernel Access for Jupyter Notebooks

**Jupyter Enterprise Gateway** empowers Jupyter Notebook users to seamlessly launch and manage remote kernels in distributed environments, opening up a world of possibilities for data science and analysis. For more in-depth information, visit the original repository: [https://github.com/jupyter-server/enterprise_gateway](https://github.com/jupyter-server/enterprise_gateway).

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

## Key Features:

*   **Remote Kernel Launch:** Launch kernels on various distributed clusters, including Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Extensive Kernel Support:** Out-of-the-box support for Python (using IPython kernel), R (using IRkernel), and Scala (using Apache Toree kernel).
*   **Flexible Deployment:**  Supports kernel launch on the Enterprise Gateway server, specific cluster nodes, or nodes managed by a resource manager.
*   **Secure Communication:** Ensures secure communication between clients, the Enterprise Gateway, and kernels.
*   **Multi-Tenant Capabilities:** Enables multiple users to utilize the gateway simultaneously.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for uninterrupted workflows.
*   **Profile Association:**  Allows associating user-specific configuration profiles with kernels.
*   **Extensible Framework:** Easily extendable to support other resource managers and kernel environments.

## Technical Overview

Jupyter Enterprise Gateway acts as a web server, providing headless access to Jupyter kernels within an enterprise environment. It offers similar functionality to Jupyter Kernel Gateway's [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html). The architecture includes remote kernel, process proxy, and launcher frameworks. For more details, refer to the [System Architecture](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html) documentation.

## Installation

Get started quickly with `pip`:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

For detailed installation instructions, including configuration options, consult the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) and [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) in the project documentation.

## Contributing

Contribute to the project by reviewing the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html) that outlines how to contribute and details the project roadmap.  Set up your development environment by following the instructions on the [development environment setup](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html) page.