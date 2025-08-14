# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels

**Jupyter Enterprise Gateway** empowers data scientists and researchers to connect Jupyter Notebooks to remote kernels in distributed computing environments, streamlining workflows and boosting productivity.  ([Original Repository](https://github.com/jupyter-server/enterprise_gateway))

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

## Key Features:

*   **Remote Kernel Support:**  Launch kernels on remote clusters like Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Broad Kernel Compatibility:** Out-of-the-box support for Python (IPython), R (IRkernel), and Scala (Apache Toree).
*   **Flexible Deployment Options:**  Supports local kernel launch, round-robin node selection, and resource manager integration.
*   **Secure Communication:** Provides secure communication between the client, Enterprise Gateway, and kernels.
*   **Multi-Tenant Capabilities:** Enables multiple users to share the gateway and access their kernels.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for continuous work.
*   **Configurable Profiles:** Associate profiles with kernels for customized settings.

[Full Documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest)

*Note: For managing multiple Jupyter Notebook deployments, consider using [JupyterHub](https://github.com/jupyterhub/jupyterhub).*

## Technical Overview

Jupyter Enterprise Gateway provides headless access to Jupyter kernels within an enterprise. It offers feature parity with Kernel Gateway's [jupyter-websocket mode](https://jupyter-kernel-gateway.readthedocs.io/en/latest/websocket-mode.html) and extends it with remote kernel management capabilities.

![Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)

## Installation

Install Jupyter Enterprise Gateway using `pip`:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

Explore the configuration options:

```bash
jupyter enterprisegateway --help-all
```

Run with default options:

```bash
jupyter enterprisegateway
```

Refer to the [configuration options within the Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway) for detailed configuration information.

## System Architecture

Learn about Enterprise Gateway's remote kernel, process proxy, and launcher frameworks on the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

## Contributing

Contribute to Jupyter Enterprise Gateway! Find out how by visiting the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html), review the [roadmap](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/roadmap.html), and set up your [development environment](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html).