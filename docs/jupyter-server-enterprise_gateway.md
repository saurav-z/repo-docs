# Jupyter Enterprise Gateway: Enable Remote Kernel Access for Jupyter Notebooks

**[GitHub Repository](https://github.com/jupyter-server/enterprise_gateway)** | **[Documentation](https://jupyter-enterprise-gateway.readthedocs.io/)**

Jupyter Enterprise Gateway empowers your Jupyter Notebook environment by enabling the launching of remote kernels in a distributed cluster.

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

Key features of Jupyter Enterprise Gateway include:

*   **Remote Kernel Support:** Launches kernels in a distributed cluster, including Apache Spark managed by YARN, IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Out-of-the-Box Kernel Support:** Provides support for Python (using IPython kernel), R (using IRkernel), and Scala (using Apache Toree kernel).
*   **Flexible Kernel Deployment:** Supports launching kernels locally, on specific cluster nodes (round-robin), or through resource managers.
*   **Secure Communication:** Ensures secure communication between the client, Enterprise Gateway server, and kernels.
*   **Multi-Tenant Capabilities:** Allows for a shared environment with isolated kernels.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for continuous operation.
*   **Extensible Framework:** Allows for the configuration of other resource managers and environments beyond the defaults.

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Install Jupyter Enterprise Gateway using pip:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

Refer to the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) for detailed installation instructions. See [configuration options within the Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) for more information.

## Technical Overview

Jupyter Enterprise Gateway provides a headless web server for accessing Jupyter kernels within an enterprise. It offers feature parity with Kernel Gateway's [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html).

## System Architecture

Learn more about the system architecture, including remote kernel, process proxy, and launcher frameworks, on the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

## Contributing

Contribute to Jupyter Enterprise Gateway by following the guidelines on the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html). Also, check out the [roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html) for future project plans. Set up a development environment using the [dev install instructions](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html).