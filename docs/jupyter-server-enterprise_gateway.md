# Jupyter Enterprise Gateway: Powering Remote Kernel Execution for Jupyter Notebooks

**Jupyter Enterprise Gateway** ([GitHub Repository](https://github.com/jupyter-server/enterprise_gateway)) unlocks the power of distributed computing by enabling Jupyter Notebooks to launch and manage kernels across remote clusters.

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

## Key Features

*   **Remote Kernel Execution:** Launch kernels on remote clusters, including Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Supported Kernels:** Out-of-the-box support for Python (IPython), R (IRkernel), and Scala (Apache Toree).
*   **Flexible Kernel Deployment:** Deploy kernels locally or on specific nodes within a cluster.
*   **Secure Communication:** Ensures secure communication between the client, Enterprise Gateway, and kernels.
*   **Multi-Tenancy:** Supports multi-tenant environments.
*   **Persistent Kernel Sessions:** Maintains persistent kernel sessions.
*   **Extensible Framework:** Easily extend to support additional resource managers.
*   **Profile Association:** Associate configuration settings with kernels for specific users (see [Project Roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html)).

## Technical Overview

Jupyter Enterprise Gateway is a web server providing headless access to Jupyter kernels within an enterprise environment. It provides feature parity with Kernel Gateway's [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html).

[Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Detailed installation instructions can be found in the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) of the project docs.

Quick Start using `pip`:

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

See the [configuration options within the Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) for supported options.

## System Architecture

Learn more about the Enterprise Gateway's remote kernel, process proxy, and launcher frameworks on the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

## Contributing

Contribute to Jupyter Enterprise Gateway! Find out how on the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html), including information on setting up a development environment and typical developer tasks.