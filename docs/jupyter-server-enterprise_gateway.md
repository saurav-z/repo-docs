# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels

Jupyter Enterprise Gateway enables seamless access to remote Jupyter kernels within a distributed cluster, empowering data scientists and engineers to leverage powerful computing resources. ([See the original repository](https://github.com/jupyter-server/enterprise_gateway))

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

**Key Features:**

*   **Remote Kernel Support:** Launch kernels on various distributed clusters like Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Out-of-the-Box Kernel Support:** Ready-to-use kernels for Python (IPython), R (IRkernel), and Scala (Apache Toree).
*   **Flexible Kernel Deployment:** Launch kernels locally or across a cluster, including utilizing resource managers.
*   **Secure Communication:** Ensures secure communication from client to kernels through the Enterprise Gateway.
*   **Multi-tenant capabilities:** Supports multiple users and projects with isolated kernel sessions.
*   **Persistent Kernel Sessions:** Maintain kernel sessions for continuous access.
*   **Extensible Framework:** Easily add support for other cluster managers.

**Comprehensive Documentation:**

*   **[Website](https://jupyter-enterprise-gateway.readthedocs.io/)**
*   **[Technical Overview](#technical-overview)**
*   **[Installation](#installation)**
*   **[System Architecture](#system-architecture)**
*   **[Contributing](#contributing)**

**Technical Overview**

Jupyter Enterprise Gateway acts as a web server, providing headless access to Jupyter kernels in an enterprise environment, inspired by the Jupyter Kernel Gateway. It extends Kernel Gateway's functionality by:

*   Supporting remote kernels hosted throughout the enterprise.
*   Providing out-of-the-box support for various cluster managers.
*   Enabling secure communication.
*   Offering multi-tenant capabilities.
*   Supporting persistent kernel sessions.
*   Offering the ability to associate profiles consisting of configuration settings to a kernel for a given user

**[Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)**

**Installation**

Detailed installation instructions are available in the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html).

Quick start using `pip`:

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

For more information on configuration options, see the [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

**System Architecture**

The [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html) provides details on Enterprise Gateway's remote kernel, process proxy, and launcher frameworks.

**Contributing**

Contribute to Jupyter Enterprise Gateway by following the guidelines on the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html), where you'll also find the project roadmap.  Set up your development environment by following the instructions on the [devinstall page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html).