# Jupyter Enterprise Gateway: Unleash the Power of Distributed Computing in Your Jupyter Notebooks

**Jupyter Enterprise Gateway** ([Original Repository](https://github.com/jupyter-server/enterprise_gateway)) empowers your Jupyter Notebooks to run kernels remotely across distributed clusters, enabling scalable and efficient data science and analysis.

[![Actions Status](https://github.com/jupyter-server/enterprise_gateway/workflows/Builds/badge.svg)](https://github.com/jupyter-server/enterprise_gateway/actions)
[![PyPI version](https://badge.fury.io/py/jupyter-enterprise-gateway.svg)](https://badge.fury.io/py/jupyter-enterprise-gateway)
[![Downloads](https://pepy.tech/badge/jupyter-enterprise-gateway/month)](https://pepy.tech/project/jupyter-enterprise-gateway)
[![Documentation Status](https://readthedocs.org/projects/jupyter-enterprise-gateway/badge/?version=latest)](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/?badge=latest)
[![Google Group](https://img.shields.io/badge/google-group-blue.svg)](https://groups.google.com/forum/#!forum/jupyter)

**Key Features:**

*   **Remote Kernel Execution:** Launch kernels on distributed clusters such as Apache Spark (managed by YARN), IBM Spectrum Conductor, Kubernetes, or Docker Swarm.
*   **Supported Kernels:** Out-of-the-box support for Python (IPython), R (IRkernel), and Scala (Apache Toree).
*   **Flexible Kernel Deployment:** Deploy kernels locally or across your cluster, using a round-robin approach or resource manager integration.
*   **Secure Communication:**  Ensures secure communication between the client, Enterprise Gateway, and kernels.
*   **Multi-tenant capabilities:** Facilitates the sharing of resources across teams.
*   **Persistent Kernel Sessions:** Allows for the persistence of kernel sessions.
*   **Extensible Framework:** Extend the Enterprise Gateway to support other resource managers.

**Learn More:**

*   **[Documentation](https://jupyter-enterprise-gateway.readthedocs.io/)**
*   **[Technical Overview](#technical-overview)**
*   **[Installation](#installation)**
*   **[System Architecture](#system-architecture)**
*   **[Contributing](#contributing)**

**Installation:**

Install Jupyter Enterprise Gateway using pip:

```bash
pip install --upgrade jupyter_enterprise_gateway
```

Detailed installation instructions are available in the [Users Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/users/index.html).

**Configuration:**

Explore the configuration options in the [Operators Guide](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/operators/index.html#configuring-enterprise-gateway).

**System Architecture:**

For a deeper dive into the architecture, see the [System Architecture page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/system-architecture.html).

**Contributing:**

Contribute to the project by following the guidelines outlined in the [Contribution page](https://jupyter-enterprise-gateway.readthedocs.io/en/latest/contributors/contrib.html).