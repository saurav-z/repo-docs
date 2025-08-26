# Backend.AI: The Open-Source Computing Cluster Platform

Backend.AI empowers researchers and developers to harness the power of containerized computing environments for machine learning, scientific computing, and more.  Check out the original repo [here](https://github.com/lablup/backend.ai).

[![PyPI release version](https://badge.fury.io/py/backend.ai-manager.svg)](https://pypi.org/project/backend.ai-manager/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/backend.ai-manager.svg)
![Wheels](https://img.shields.io/pypi/wheel/backend.ai-manager.svg)
[![Gitter](https://badges.gitter.im/lablup/backend.ai.svg)](https://gitter.im/lablup/backend.ai)

## Key Features

*   **Container-Based Computing:**  Easily deploy and manage computing environments using containers, streamlining your workflow.
*   **Framework & Language Support:**  Supports popular computing/ML frameworks and diverse programming languages, offering flexibility for various projects.
*   **Accelerator Support:**  Includes pluggable support for heterogeneous accelerators like CUDA GPUs, ROCm GPUs, and more for optimized performance.
*   **Resource Management:**  Efficiently allocates and isolates computing resources for multi-tenant sessions, ensuring optimal utilization.
*   **API-Driven:**  All functionalities are accessible via REST and GraphQL APIs, facilitating easy integration and automation.
*   **Customizable Scheduling:** Leverage customizable job schedulers with "Sokovan" orchestrator for batch and on-demand computations.

## Core Components

*   **Manager:** The cluster control-plane, responsible for routing API requests, monitoring agents, and scaling the cluster.
*   **Agent:**  Manages individual server instances, launches/destroys Docker containers, and runs REPL daemons (kernels).
*   **Storage Proxy:** Provides a unified abstraction over multiple network storage devices, offering performance metrics and acceleration APIs.
*   **Webserver:** Hosts the SPA (single-page application) web UI for end-users and administration tasks.
*   **Kernels:**  Computing environment recipes (Dockerfile) for building container images.

## Getting Started

### Installation for Single-node Development

1.  Clone the repository.
2.  Run `./scripts/install-dev.sh`. This script requires `sudo` and a modern Python installation on Linux or macOS.

### Accessing Compute Sessions (Kernels)

Backend.AI offers several ways to access your computation sessions:

*   **Jupyter:**  Built-in support for Jupyter and JupyterLab.
*   **Web-based terminal:** Intrinsic ttyd support in all container sessions.
*   **SSH:** SSH/SFTP/SCP support with auto-generated SSH keypairs, enabling integration with IDEs like PyCharm and VSCode.
*   **VSCode:** Web-based VSCode support in most container sessions.

### Working with Storage

Backend.AI provides "vfolders" (virtual folders) that abstract network-based storage devices (e.g., NFS/SMB). These can be mounted into computation sessions and shared among users with different privileges.

## Client SDKs

*   **Python:** `pip install backend.ai-client`
*   **Java:**  Available via GitHub releases
*   **Javascript:** `npm install backend.ai-client`
*   **PHP:** (Under preparation)

## Plugins

Extend Backend.AI with plugins:

*   `backendai_accelerator_v21`: Accelerator plugins (CUDA, ROCm, etc.)
*   `backendai_monitor_stats_v10`: Statistics collectors (e.g., Datadog)
*   `backendai_monitor_error_v10`: Exception collectors (e.g., Sentry)

## Python Version Compatibility

| Backend.AI Core Version | Python Version | Pantsbuild version |
|:-----------------------:|:--------------:|:------------------:|
| 25.06.x ~               | 3.13.x         | 2.23.x             |
| 24.03.x / 24.09.x ~ 25.05.x      | 3.12.x         | 2.21.x    |
| 23.03.x / 23.09.x       | 3.11.x         | 2.19.x             |
| 22.03.x / 22.09.x       | 3.10.x         |                    |
| 21.03.x / 21.09.x       | 3.8.x          |                    |

## Building Packages

Build Python wheels and SCIE (Self-Contained Installable Executables):

```bash
./scripts/build-wheels.sh
./scripts/build-scies.sh
```

Built packages are placed in the `dist/` directory.

## License

Refer to the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE).