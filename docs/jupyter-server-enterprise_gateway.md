# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels for Jupyter Notebooks

**[GitHub Repository](https://github.com/jupyter-server/enterprise_gateway)** | **[Documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/)**

Jupyter Enterprise Gateway empowers your Jupyter Notebooks to connect with remote kernels in a distributed cluster, enabling scalable and efficient data science and analysis.

**Key Features:**

*   **Remote Kernel Support:** Launches kernels on remote nodes, including clusters managed by YARN, IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Out-of-the-Box Kernel Support:** Supports popular kernels like Python (IPython), R (IRkernel), and Scala (Apache Toree).
*   **Secure Communication:** Ensures secure communication between the client, Enterprise Gateway, and the kernels.
*   **Multi-Tenant Capabilities:** Facilitates multi-user environments.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for continuous workflows.
*   **Extensible Framework:** Allows for configuration and support of various resource managers and kernel environments.
*   **[Deployment Diagram](https://github.com/jupyter-server/enterprise_gateway/blob/main/docs/source/images/deployment.png?raw=true)**

**Getting Started:**

Install Jupyter Enterprise Gateway using pip:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

Explore configuration options:

```bash
jupyter enterprisegateway --help-all
```

Run with default options:

```bash
jupyter enterprisegateway
```

For detailed installation instructions and configuration options, refer to the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) and [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

**Dive Deeper:**

*   **System Architecture:** Learn about the inner workings of Enterprise Gateway, including its remote kernel, process proxy, and launcher frameworks: [System Architecture](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html)
*   **Contributing:** Discover how to contribute to the project: [Contribution Page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html).