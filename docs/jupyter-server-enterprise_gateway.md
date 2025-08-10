# Jupyter Enterprise Gateway: Unlock Scalable & Secure Remote Kernel Access

**[Visit the Official Website](https://jupyter-enterprise-gateway.readthedocs.io/)** |
**[Explore the Technical Overview](#technical-overview)** |
**[Get Started with Installation](#installation)** |
**[Understand the System Architecture](#system-architecture)** |
**[Contribute to the Project](#contributing)**

Jupyter Enterprise Gateway empowers your Jupyter Notebook environment to connect with and manage remote kernels in distributed clusters, enabling powerful data science and analysis at scale.  ([View the original repository on GitHub](https://github.com/jupyter-server/enterprise_gateway))

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

**Key Features of Jupyter Enterprise Gateway:**

*   **Remote Kernel Launching:**  Connect to kernels hosted across your enterprise, including those managed by YARN, IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Broad Kernel Support:** Out-of-the-box support for Python (via IPython), R (via IRkernel), and Scala (via Apache Toree).
*   **Secure Communication:** Ensures secure communication between the client, Enterprise Gateway, and kernels.
*   **Multi-Tenancy:**  Supports multiple users and projects within a single Enterprise Gateway instance.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for uninterrupted workflows.
*   **Configurable Profiles:** Allows associating profiles with specific kernel configurations for each user.
*   **Extensible Framework:**  Offers an extensible framework to integrate with other resource managers.

## Technical Overview

Jupyter Enterprise Gateway provides a robust solution for accessing Jupyter kernels within an enterprise environment, building upon the functionality of Jupyter Kernel Gateway.  It extends this functionality by:

*   Offering the option to launch kernels locally or on specific nodes in your cluster, including a round-robin algorithm.
*   Providing built-in support for Apache Spark managed by YARN, IBM Spectrum Conductor, Kubernetes or Docker Swarm.
*   Supporting Multi-tenant capabilities
*   Persistence kernel sessions
*   Profiles configuration support

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Get started quickly with `pip`:

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

For detailed installation instructions and configuration options, please refer to the [User Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) and the [Operator's Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) within the project documentation.

## System Architecture

Dive deeper into the architecture of Enterprise Gateway, including its remote kernel, process proxy, and launcher frameworks, by visiting the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

## Contributing

We welcome contributions!  Learn how to contribute, including setting up your development environment and exploring developer tasks, by visiting the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html) and reviewing our roadmap.