# Jupyter Enterprise Gateway: Empowering Remote Kernel Execution for Jupyter Notebooks

Jupyter Enterprise Gateway enables scalable and secure access to remote Jupyter kernels within a distributed computing environment, allowing users to harness the power of remote resources. [(See the original repository)](https://github.com/jupyter-server/enterprise_gateway)

**Key Features:**

*   **Remote Kernel Launch:** Enables Jupyter Notebook to launch kernels on remote clusters, including Apache Spark managed by YARN, IBM Spectrum Conductor, Kubernetes, and Docker Swarm.
*   **Supported Kernels:** Provides out-of-the-box support for Python (using IPython kernel), R (using IRkernel), and Scala (using Apache Toree kernel).
*   **Flexible Deployment:** Supports launching kernels locally, on specific cluster nodes (using a round-robin algorithm), or on nodes managed by resource managers.
*   **Secure Communication:** Ensures secure communication between the client, the Enterprise Gateway server, and the kernels.
*   **Multi-tenancy & Persistent Sessions:** Provides multi-tenant capabilities and persistent kernel sessions for enhanced user experience.
*   **Extensible Framework:** Offers an extensible framework to support other kernel types and resource managers.

**Technical Overview:**

Jupyter Enterprise Gateway extends the functionality of Jupyter Kernel Gateway, providing headless access to Jupyter kernels within an enterprise. It supports remote kernel hosting and offers features such as:

*   Support for remote kernels.
*   Support for Apache Spark managed by YARN, IBM Spectrum Conductor, Kubernetes or Docker Swarm.
*   Multi-tenant capabilities
*   Persistent kernel sessions
*   Ability to associate profiles consisting of configuration settings to a kernel for a given user

**Installation:**

Install Jupyter Enterprise Gateway using pip:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

For detailed installation and configuration instructions, see the [User's Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html) and [Operator's Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

**System Architecture:**

For detailed information about Enterprise Gateway's remote kernel, process proxy, and launcher frameworks, please refer to the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

**Contributing:**

Contribute to Jupyter Enterprise Gateway by following the guidelines on the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html), including the [development environment setup](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/devinstall.html).