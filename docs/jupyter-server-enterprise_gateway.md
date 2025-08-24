# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels in Your Enterprise

[<img src="https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg" alt="Build Status">](https://github.com/jupyter-server/enterprise_gateway/actions)
[<img src="https://badge.fury.io/py/jupyter-enterprise-gateway.svg" alt="PyPI version">](https://pypi.org/project/jupyter-enterprise-gateway/)
[<img src="https://pepy.tech/badge/jupyter-enterprise-gateway/month" alt="Downloads">](https://pepy.tech/project/jupyter-enterprise-gateway)
[<img src="https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest" alt="Documentation Status">](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[<img src="https://img.shields.io/badge/google-group-blue.svg" alt="Google Group">](https://groups.google.com/forum/#!forum/jupyter)

Jupyter Enterprise Gateway empowers data scientists and engineers to harness the full potential of distributed computing by seamlessly connecting Jupyter Notebooks to remote kernels across your enterprise.

**Key Features:**

*   **Remote Kernel Support:** Launch kernels on remote nodes within a cluster.
*   **Cluster Integration:** Out-of-the-box support for Apache Spark (YARN), IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Secure Communication:** Securely connects clients to remote kernels.
*   **Multi-Tenancy:** Supports multiple users and projects.
*   **Persistent Sessions:** Maintains kernel sessions for continuous work.
*   **Extensible Framework:** Easily configure support for other resource managers and kernel types.
*   **Supported Kernels:**
    *   Python (IPython kernel)
    *   R (IRkernel)
    *   Scala (Apache Toree kernel)

**Getting Started**

1.  **Installation:**

    ```bash
    pip install --upgrade jupyter_enterprise_gateway
    ```

2.  **Configuration:**

    *   View all configuration options: `jupyter enterprisegateway --help-all`
    *   Run with default options: `jupyter enterprisegateway`

3.  **Documentation:**
    *   For detailed installation and configuration instructions, refer to the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html).
    *   Explore configuration options in the [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

**Learn More**

*   **[Website](https://jupyter-enterprise-gateway.readthedocs.io/)**
*   **[Technical Overview](#technical-overview)**
*   **[System Architecture](#system-architecture)**
*   **[Contributing](#contributing)**
*   **Original Repository:** [https://github.com/jupyter-server/enterprise_gateway](https://github.com/jupyter-server/enterprise_gateway)

For managing multiple Jupyter Notebook deployments, consider using [JupyterHub](https://github.com/jupyterhub/jupyterhub).

**Technical Overview:**

Jupyter Enterprise Gateway provides a headless access to Jupyter kernels.

**System Architecture:**

[View the System Architecture](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html) to understand the remote kernel, process proxy, and launcher frameworks.

**Contributing:**

Learn how to contribute to Jupyter Enterprise Gateway, including setting up a development environment and developer tasks, at the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html).