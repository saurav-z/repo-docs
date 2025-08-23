# Jupyter Enterprise Gateway: Empowering Remote Kernel Access for Jupyter Notebooks

**[Website](https://jupyter-enterprise-gateway.readthedocs.io/)** |
**[Documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest)** |
**[Installation](#installation)** |
**[System Architecture](#system-architecture)** |
**[Contributing](#contributing)**

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

Jupyter Enterprise Gateway unlocks the power of distributed computing, allowing you to run Jupyter Notebook kernels on remote clusters.

**Key Features:**

*   **Remote Kernel Launching:** Launch kernels on various cluster environments, including Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Flexible Deployment:** Supports launching kernels locally or on specific nodes within your cluster, leveraging resource managers.
*   **Secure Communication:** Provides secure communication between the client, Enterprise Gateway, and the kernels.
*   **Multi-Tenant Capabilities:** Supports a multi-tenant environment for efficient resource utilization.
*   **Persistent Kernel Sessions:** Maintains persistent kernel sessions for uninterrupted workflows.
*   **Extensible Framework:** Easily configure Enterprise Gateway to support other resource managers beyond those provided out of the box.
*   **Out-of-the-box Kernel Support:** Offers pre-configured support for Python (IPython), R (IRkernel), and Scala (Apache Toree kernel).

**Learn more about the power of remote kernels at the [official repository](https://github.com/jupyter-server/enterprise_gateway).**

For a deeper dive, please consult the comprehensive documentation:

*   [Full Documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest)
*   [System Architecture](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html)
*   [Contribution Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html)

**Installation**

Detailed installation instructions are available in the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html).  Here's a quick start using `pip`:

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

Check the [configuration options within the Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) for a complete overview of supported options.

**System Architecture**

The [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html) details the Enterprise Gateway's remote kernel, process proxy, and launcher frameworks.

**Contributing**

The [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html) explains how to contribute to Enterprise Gateway and provides access to the project roadmap.  You can also find instructions on [setting up a development environment](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html).