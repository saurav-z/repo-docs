# Jupyter Enterprise Gateway: Powering Remote Kernel Access for Jupyter Notebooks

**Jupyter Enterprise Gateway** empowers Jupyter Notebook users to seamlessly launch and manage remote kernels in distributed environments, streamlining data science and analysis workflows.  [View the project on GitHub](https://github.com/jupyter-server/enterprise_gateway).

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise_gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

**Key Features:**

*   **Remote Kernel Launching:** Launch kernels on distributed clusters like Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Broad Kernel Support:** Out-of-the-box support for Python (IPython), R (IRkernel), and Scala (Apache Toree).
*   **Flexible Deployment:**  Supports launching kernels locally, on specific cluster nodes (round-robin), or via resource managers.
*   **Secure Communication:** Ensures secure communication between the client, Enterprise Gateway, and kernels.
*   **Multi-Tenant Capabilities:** Enables multiple users to leverage the gateway simultaneously.
*   **Persistent Kernel Sessions:**  Maintains kernel sessions for uninterrupted workflows.
*   **Configurable Profiles:** Allows association of custom profiles with kernel configurations per user (see the [Project Roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html)).

For managing multiple Jupyter Notebook deployments, consider using [JupyterHub](https://github.com/jupyterhub/jupyterhub).

**[Full Documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/)**

## Technical Overview

Jupyter Enterprise Gateway extends the functionality of Jupyter Kernel Gateway by providing headless access to Jupyter kernels, with support for remote kernel hosting and advanced features:

*   **Remote Kernel Management:** Facilitates launching kernels on different nodes of the cluster.
*   **Resource Manager Integration:**  Provides built-in support for common cluster managers.
*   **Extensible Framework:** Allows for easy integration with other resource managers.

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Detailed installation instructions are available in the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html).

Quickstart using `pip`:

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

Explore the [configuration options](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) in the Operators Guide.

## System Architecture

Learn more about the underlying architecture, including remote kernel, process proxy, and launcher frameworks, on the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

## Contributing

Contribute to Jupyter Enterprise Gateway!  Refer to the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html) for guidelines and the project roadmap.  Set up your [development environment](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html) and explore common developer tasks.