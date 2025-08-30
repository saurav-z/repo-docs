# Jupyter Enterprise Gateway: Unleash the Power of Distributed Computing for Jupyter Notebooks

[<img src="https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg" alt="Build Status">](https://github.com/jupyter-server/enterprise_gateway/actions)
[<img src="https://badge.fury.io/py/jupyter-enterprise-gateway.svg" alt="PyPI version">](https://badge.fury.io/py/jupyter-enterprise-gateway)
[<img src="https://pepy.tech/badge/jupyter-enterprise-gateway/month" alt="Downloads">](https://pepy.tech/project/jupyter-enterprise-gateway)
[<img src="https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest" alt="Documentation Status">](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[<img src="https://img.shields.io/badge/google-group-blue.svg" alt="Google Group">](https://groups.google.com/forum/#!forum/jupyter)

Jupyter Enterprise Gateway empowers data scientists and engineers to leverage the full potential of distributed computing environments directly from their Jupyter Notebooks.

**[Website](https://jupyter-enterprise-gateway.readthedocs.io/)** |
**[Technical Overview](#technical-overview)** |
**[Installation](#installation)** |
**[System Architecture](#system-architecture)** |
**[Contributing](#contributing)**

## Key Features

*   **Remote Kernel Execution:** Launch and manage Jupyter kernels on remote clusters and nodes.
*   **Cluster Support:** Seamlessly integrates with Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Extensible Architecture:** Supports custom resource managers through a flexible and extensible framework.
*   **Secure Communication:** Provides secure communication between the client, the Enterprise Gateway, and the kernels.
*   **Multi-Tenant Capabilities:** Enables the sharing of resources across multiple users.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for uninterrupted workflow.
*   **Profile Management:**  Associate custom configurations with kernels for different users.
*   **Out-of-the-Box Kernel Support:** Includes support for Python (with IPython kernel), R (with IRkernel), and Scala (with Apache Toree kernel).

## Technical Overview

Jupyter Enterprise Gateway acts as a web server that provides headless access to Jupyter kernels in an enterprise environment.  It builds upon the functionality of Jupyter Kernel Gateway, offering feature parity with its [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html) while adding significant enhancements for distributed computing.

[Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Install Jupyter Enterprise Gateway quickly using `pip`:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

For detailed installation instructions, please refer to the [User Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html). Explore configuration options in the [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) and see all available configuration options with `jupyter enterprisegateway --help-all`.

## System Architecture

Learn more about the inner workings of Enterprise Gateway, including its remote kernel, process proxy, and launcher frameworks, on the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

## Contributing

Contribute to the project and help shape the future of Jupyter Enterprise Gateway!  See the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html) for details on how to contribute, including setting up a development environment and typical developer tasks. Check out the [Project Roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html) to see upcoming features.

**[Back to the original repository](https://github.com/jupyter-server/enterprise_gateway)**