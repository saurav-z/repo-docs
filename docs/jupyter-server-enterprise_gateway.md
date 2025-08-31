# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels

**Jupyter Enterprise Gateway** empowers your Jupyter Notebook environment by enabling the launch and management of remote kernels in distributed clusters, fostering collaboration and scalability.  Learn more about the project on the [Jupyter Enterprise Gateway GitHub](https://github.com/jupyter-server/enterprise_gateway).

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

## Key Features

Jupyter Enterprise Gateway offers a robust set of features for managing remote kernels in an enterprise setting:

*   **Remote Kernel Support:** Launch kernels on local machines, specific cluster nodes (using round-robin), or through resource managers.
*   **Cluster Integration:** Seamlessly integrates with Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Secure Communication:** Ensures secure communication between clients, the Gateway server, and kernels.
*   **Multi-Tenant Capabilities:** Supports multiple users and environments.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for continuous operation.
*   **Profile Management:** Allows association of configuration profiles with kernels for customized user experiences.
*   **Out-of-the-Box Kernel Support:** Supports Python (IPython kernel), R (IRkernel), and Scala (Apache Toree kernel).

For comprehensive information, refer to the [full documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest).

## Installation

Get started with Jupyter Enterprise Gateway using `pip`:

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

Detailed installation and configuration instructions can be found in the [User Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) and [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

## Technical Overview

The [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html) provides an in-depth look at the architecture, including the remote kernel, process proxy, and launcher frameworks.

## Contributing

Contribute to the project by following the guidelines on the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html), including setting up a development environment ([devinstall](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html)) and common developer tasks.