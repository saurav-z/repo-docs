# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels for Jupyter Notebooks

**Jupyter Enterprise Gateway** ([GitHub Repository](https://github.com/jupyter-server/enterprise_gateway)) empowers your Jupyter Notebooks to harness the computational power of distributed clusters, enabling seamless access to remote kernels for enhanced data science and machine learning workflows.

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

## Key Features

*   **Remote Kernel Launching:** Launch kernels on distributed clusters like Apache Spark managed by YARN, IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Broad Kernel Support:** Supports popular kernels out-of-the-box:
    *   Python (IPython kernel)
    *   R (IRkernel)
    *   Scala (Apache Toree kernel)
*   **Flexible Deployment Options:** Launch kernels locally or on specific cluster nodes using a round-robin algorithm or through a resource manager.
*   **Secure Communication:** Ensures secure communication from the client through the Enterprise Gateway server to the kernels.
*   **Multi-tenant Capabilities:** Supports multi-tenant environments for efficient resource utilization.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for uninterrupted workflows.
*   **Extensible Framework:** Easily configurable to support other resource managers via Enterprise Gateway's extensible framework.
*   **Profile Management:** Associate profiles with kernel configurations for specific users (see [Project Roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html)).

## Technical Overview

Jupyter Enterprise Gateway acts as a web server that provides headless access to Jupyter kernels within an enterprise. It mirrors features of Jupyter Kernel Gateway's [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html) while expanding on capabilities for remote kernel management.  For a detailed understanding, refer to the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Install Jupyter Enterprise Gateway easily using `pip`:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

For detailed installation instructions and configuration options, please consult the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) and the [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

## Contributing

We welcome contributions! Learn how to contribute to Jupyter Enterprise Gateway, set up your development environment, and review the project roadmap on the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html).