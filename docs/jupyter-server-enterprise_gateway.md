# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels for Jupyter Notebooks

**Jupyter Enterprise Gateway** ([Original Repo](https://github.com/jupyter-server/enterprise_gateway)) empowers Jupyter Notebook users to leverage remote kernels in distributed environments, enhancing scalability and resource utilization.

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

Key features of Jupyter Enterprise Gateway:

*   **Remote Kernel Support:** Launches and manages Jupyter kernels across a distributed cluster, including environments like Apache Spark managed by YARN, IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Enhanced Kernel Management:** Offers features like secure communication, multi-tenancy, and persistent kernel sessions.
*   **Flexible Deployment:** Supports kernel launching locally or on specific cluster nodes, including those managed by resource managers.
*   **Out-of-the-Box Kernel Support:** Includes support for Python (IPython kernel), R (IRkernel), and Scala (Apache Toree kernel).
*   **Extensible Framework:** Allows for customization and extension to support additional kernel types and resource managers.
*   **Configuration Profiles:** Enables associating profiles with kernels for customized configurations per user.

**Full Documentation:** [Jupyter Enterprise Gateway Documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest)

For deploying and managing multiple Jupyter Notebook deployments, consider [JupyterHub](https://github.com/jupyterhub/jupyterhub).

## Technical Overview

Jupyter Enterprise Gateway builds upon Jupyter Kernel Gateway, providing feature parity with its [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html) and extending its capabilities.  It enables secure communication from client to kernels through the Enterprise Gateway server. It also allows for multi-tenant capabilities and persistent kernel sessions.

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Install Jupyter Enterprise Gateway using `pip`:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

For detailed installation instructions, refer to the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html).

**Configuration:** Use `jupyter enterprisegateway --help-all` to view all configuration options and see the [configuration options within the Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) for information about supported options.

## System Architecture

Learn more about the components of Jupyter Enterprise Gateway on the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

## Contributing

Contribute to the project by following the guidelines on the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html).  Set up a development environment as described in the [development installation instructions](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html) and explore common developer tasks.