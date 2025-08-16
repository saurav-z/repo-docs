# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels

**Jupyter Enterprise Gateway** empowers your Jupyter Notebooks to harness the power of remote kernels in distributed computing environments, enabling scalable data science and analysis.  [Explore the project on GitHub](https://github.com/jupyter-server/enterprise_gateway).

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

**Key Features:**

*   **Remote Kernel Support:** Launch and manage kernels across a distributed cluster, including Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Versatile Kernel Support:** Works seamlessly with Python (IPython), R (IRkernel), and Scala (Apache Toree kernel).
*   **Enhanced Security:** Secure communication between clients, the Enterprise Gateway server, and kernels.
*   **Multi-Tenancy:** Supports multiple users and environments.
*   **Persistent Kernel Sessions:** Maintain kernel sessions for continuous work.
*   **Extensible Framework:** Easily integrate with other resource managers and cluster technologies.
*   **Profile Management:** Associate profiles with configuration settings to customize kernel behavior.

For deployments that require multiple Jupyter Notebook instances, consider using [JupyterHub](https://github.com/jupyterhub/jupyterhub).

## Technical Overview

Jupyter Enterprise Gateway extends Jupyter Kernel Gateway, providing headless access to Jupyter kernels with added functionality:

*   Supports remote kernel hosting.
*   Integrates with resource managers.
*   Offers multi-tenant capabilities.

## Installation

Get started quickly using `pip`:

```bash
pip install --upgrade jupyter_enterprise_gateway
jupyter enterprisegateway --help-all
jupyter enterprisegateway
```

Detailed installation instructions are available in the [User Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html). Review [configuration options within the Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

## System Architecture

Learn more about Enterprise Gateway's remote kernel, process proxy, and launcher frameworks on the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

## Contributing

Join the community!  The [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html) outlines how to contribute and includes the project roadmap.  Set up your development environment, and learn about common developer tasks by exploring the [devinstall page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html).