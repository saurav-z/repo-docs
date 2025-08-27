# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels for Jupyter Notebook

**[Website](https://jupyter-enterprise-gateway.readthedocs.io/)** | **[Documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/)** | **[Installation](#installation)** | **[System Architecture](#system-architecture)** | **[Contributing](#contributing)**

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

**Jupyter Enterprise Gateway empowers data scientists and engineers to leverage remote kernels in distributed computing environments, enhancing the capabilities of Jupyter Notebook.**

This powerful tool enables seamless access to kernels hosted in clusters, offering unparalleled flexibility and scalability for your data science workflows.

**Key Features:**

*   **Remote Kernel Launch:**  Launch Jupyter kernels on remote clusters, including Apache Spark managed by YARN, IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Supported Kernels Out-of-the-Box:** Works with Python (IPython kernel), R (IRkernel), and Scala (Apache Toree kernel).
*   **Enhanced Security:** Provides secure communication between the client, Enterprise Gateway, and kernels.
*   **Multi-Tenant Support:** Allows multiple users to share and utilize the gateway.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for uninterrupted work.
*   **Extensible Framework:** Easily configure support for additional resource managers.
*   **Profile Management:** Associate profiles with kernel configurations for individual users.

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Install Jupyter Enterprise Gateway easily using pip:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

Detailed installation instructions and configuration options can be found in the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) and [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

## System Architecture

Learn more about the architecture and inner workings of Enterprise Gateway in the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html), including details on the remote kernel, process proxy, and launcher frameworks.

## Contributing

We welcome contributions!  Review the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html) for information on how to contribute, the project roadmap, and how to [set up a development environment](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html).

**[View the source code on GitHub](https://github.com/jupyter-server/enterprise_gateway)**