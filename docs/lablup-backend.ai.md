# Backend.AI: Your All-in-One Platform for Containerized Computing and AI Workloads

[Backend.AI](https://github.com/lablup/backend.ai) is a powerful platform that simplifies and accelerates your computing and AI/ML workloads with container-based infrastructure and robust features.

[![PyPI release version](https://badge.fury.io/py/backend.ai-manager.svg)](https://pypi.org/project/backend.ai-manager/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/backend.ai-manager.svg)
![Wheels](https://img.shields.io/pypi/wheel/backend.ai-manager.svg)
[![Gitter](https://badges.gitter.im/lablup/backend.ai.svg)](https://gitter.im/lablup/backend.ai)

**Key Features:**

*   **Container-Based Computing:** Easily deploy and manage containerized computing environments.
*   **Framework & Language Support:** Hosts popular computing/ML frameworks and diverse programming languages.
*   **Heterogeneous Accelerator Support:** Includes support for CUDA GPU, ROCm GPU, and other NPUs for accelerated computing.
*   **Multi-Tenant & Resource Isolation:** Allocates and isolates resources for secure, multi-tenant computation sessions.
*   **REST and GraphQL APIs:** Exposes all functionalities through standardized APIs for easy integration.
*   **Flexible Job Scheduling:** Customizable job schedulers, including the built-in "Sokovan" orchestrator.
*   **Web-Based Access:** Provides web-based access to computing sessions via Jupyter, web terminals, SSH, and VS Code.
*   **Storage Abstraction:** Simplifies storage operations with vfolders for easy data sharing and access.
*   **Client SDKs:**  Python, Java, JavaScript, and PHP SDKs are available to simplify integration.

## Components

### Manager

The Manager is the cluster control plane that routes external API requests and manages the cluster.

### Agent

Manages individual server instances and launches/destroys Docker containers.

### Storage Proxy

Provides a unified abstraction over network storage devices.

### Webserver

Hosts the web UI for end-users and administration.

##  Getting Started

*   **Installation for Single-node Development:** Run `scripts/install-dev.sh` after cloning this repository. This requires `sudo` and a modern Linux (Debian/RHEL-likes) or macOS installation.
*   **Installation for Multi-node Tests &amp; Production:** Consult [our documentation](http://docs.backend.ai) or contact our sales team (contact@lablup.com).

## Accessing Compute Sessions (Kernels)

Backend.AI offers flexible methods for accessing your computing sessions:

*   **Jupyter:**  Integrates data science workflows with built-in Jupyter and JupyterLab support.
*   **Web-based terminal:** Access sessions through a web-based terminal (ttyd).
*   **SSH:** Securely connect to sessions via SSH/SFTP/SCP with auto-generated SSH keypairs. Supports IDEs like PyCharm.
*   **VSCode:** Provides web-based VSCode support within the containers.

## Plugins

Extend Backend.AI with plugins using Python package entrypoints. Examples include:

*   `backendai_accelerator_v21`:
    *   `ai.backend.accelerator.cuda`: CUDA accelerator plugin
    *   `ai.backend.accelerator.cuda` (mock): CUDA mockup plugin
    *   `ai.backend.accelerator.rocm`
*   `backendai_monitor_stats_v10`: Datadog statistics collector.
*   `backendai_monitor_error_v10`: Sentry exception collector.

## Legacy Components

These components are still available but no longer actively maintained:

*   **Media:** Front-end libraries for multi-media output.
*   **IDE and Editor Extensions:**  VSCode and Atom extensions (recommend using in-kernel applications like Jupyter Lab, VS Code Server, or SSH connections).

## Python Version Compatibility

| Backend.AI Core Version | Python Version | Pantsbuild version |
|:-----------------------:|:--------------:|:------------------:|
| 25.06.x ~               | 3.13.x         | 2.23.x             |
| 24.03.x / 24.09.x ~ 25.05.x      | 3.12.x         | 2.21.x    |
| 23.03.x / 23.09.x       | 3.11.x         | 2.19.x             |
| 22.03.x / 22.09.x       | 3.10.x         |                    |
| 21.03.x / 21.09.x       | 3.8.x          |                    |

## Building Packages

Backend.AI packages can be built as:

1.  Python wheels (.whl)
2.  SCIE (Self-Contained Installable Executables)

Build packages with:

```bash
./scripts/build-wheel.sh
./scripts/build-scies.sh
```

## License

Refer to the [LICENSE](https://github.com/lablup/backend.ai/blob/main/LICENSE) file.