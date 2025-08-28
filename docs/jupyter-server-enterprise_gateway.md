# Jupyter Enterprise Gateway: Unleash the Power of Remote Kernels in Your Enterprise

Jupyter Enterprise Gateway empowers your Jupyter Notebook environment by enabling remote kernel execution across distributed clusters like Apache Spark, Kubernetes, and more.

**[Visit the Jupyter Enterprise Gateway Repository on GitHub](https://github.com/jupyter-server/enterprise_gateway)**

**Key Features:**

*   **Remote Kernel Execution:** Launch kernels on remote nodes within your cluster, supporting environments like YARN, Kubernetes, and Docker Swarm.
*   **Broad Kernel Support:** Out-of-the-box support for Python (IPython), R (IRkernel), and Scala (Apache Toree).
*   **Secure Communication:** Ensures secure communication between clients, the gateway, and kernels.
*   **Multi-Tenancy:** Supports multi-tenant environments, enabling efficient resource utilization.
*   **Persistent Kernel Sessions:** Maintains kernel sessions for uninterrupted workflows.
*   **Extensible Framework:** Allows you to configure support for additional resource managers beyond the default options.
*   **Profile Management:** Associate custom configurations and settings to your kernel sessions.

**Key Components:**

*   **Remote Kernel:** Enables Jupyter Notebook to connect to remote kernels, providing the compute resources for kernel execution.
*   **Process Proxy:** Manages communication and resource allocation to enable remote kernel operation.
*   **Launcher Framework:** Provides a modular system for launching kernels on different platforms and resource managers.

**Installation**

Detailed installation instructions are available in the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html).

Quick start with pip:

```bash
# install from pypi
pip install --upgrade jupyter_enterprise_gateway

# show all config options
jupyter enterprisegateway --help-all

# run it with default options
jupyter enterprisegateway
```

For additional configuration options, please consult the [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

**Additional Resources:**

*   **[Technical Overview](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html)**
*   **[Contributing](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html)**
*   **[Documentation](https://jupyter-enterprise-gateway.readthedocs.io/en/latest)**