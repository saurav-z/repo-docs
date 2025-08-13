# Backend.AI: Your Containerized Computing Platform for AI and HPC

[![PyPI release version](https://badge.fury.io/py/backend.ai-manager.svg)](https://pypi.org/project/backend.ai-manager/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/backend.ai-manager.svg)
![Wheels](https://img.shields.io/pypi/wheel/backend.ai-manager.svg)
[![Gitter](https://badges.gitter.im/lablup/backend.ai.svg)](https://gitter.im/lablup/backend.ai)

**Backend.AI** is a powerful, container-based platform designed to streamline computing, machine learning, and high-performance computing (HPC) workloads.  It hosts popular computing/ML frameworks, supports diverse programming languages, and provides pluggable accelerator support.  [Explore the Backend.AI GitHub Repository](https://github.com/lablup/backend.ai) to learn more.

## Key Features

*   **Container-Based:** Leverages containers for consistent and isolated environments.
*   **Framework & Language Support:** Hosts popular computing/ML frameworks and a variety of programming languages.
*   **Heterogeneous Accelerator Support:** Integrates with CUDA GPUs, ROCm GPUs, and other NPUs.
*   **Resource Management:** On-demand or batch computation sessions with customizable job schedulers using the Sokovan orchestrator.
*   **API Driven:** All functionalities are exposed via REST and GraphQL APIs.
*   **Flexible Access:** Provides websocket tunneling, Jupyter, web-based terminal, SSH and VSCode access to compute sessions.
*   **Storage Abstraction:**  Offers vfolders for managing and sharing data.
*   **Client SDKs:** SDKs available in Python, Java, JavaScript, and PHP.

## Core Components

### Manager

*   Routes external API requests.
*   Monitors and scales the cluster.
*   Plugin interfaces available for: `backendai_scheduler_v10`, `backendai_agentselector_v10`, `backendai_hook_v20`, `backendai_webapp_v20`, `backendai_monitor_stats_v10`, and `backendai_monitor_error_v10`.

### Agent

*   Manages server instances.
*   Launches/destroys Docker containers.
*   Plugin interfaces available for: `backendai_accelerator_v21`, `backendai_monitor_stats_v10`, and `backendai_monitor_error_v10`.

### Storage Proxy

*   Provides unified abstraction over network storage devices.

### Webserver

*   Hosts the WebUI for end-users and administration.

## Other Key Components

*   **Kernels:** Computing environment recipes (Dockerfiles) for container images.
*   **Jail:** A programmable sandbox for system call filtering.
*   **Hook:** A set of libc overrides for resource control.
*   **Client SDK Libraries:** For Python, Java, JavaScript, and PHP (under preparation).

## Plugins

Backend.AI supports plugin extensions through Python package entrypoints, and the following are mainly used:

*   **Accelerator Plugins:**  `backendai_accelerator_v21` (CUDA, ROCm, etc.).
*   **Monitoring Plugins:** `backendai_monitor_stats_v10` and `backendai_monitor_error_v10` (Datadog, Sentry).

## Getting Started

### Installation for Single-node Development

Run `scripts/install-dev.sh` after cloning the repository.

### Installation for Multi-node Tests &amp; Production

Please consult [our documentation](http://docs.backend.ai).

### Accessing Compute Sessions (aka Kernels)

Backend.AI provides websocket tunneling into individual computation sessions (containers),
so that users can use their browsers and client CLI to access in-container applications directly
in a secure way.

* Jupyter: data scientists' favorite tool
   * Most container images have intrinsic Jupyter and JupyterLab support.
* Web-based terminal
   * All container sessions have intrinsic ttyd support.
* SSH
   * All container sessions have intrinsic SSH/SFTP/SCP support with auto-generated per-user SSH keypair.
     PyCharm and other IDEs can use on-demand sessions using SSH remote interpreters.
* VSCode
   * Most container sessions have intrinsic web-based VSCode support.

### Working with Storage

Backend.AI provides an abstraction layer on top of existing network-based storages
(e.g., NFS/SMB), called vfolders (virtual folders).
Each vfolder works like a cloud storage that can be mounted into any computation
sessions and shared between users and user groups with differentiated privileges.

## Building Packages

Backend.AI supports building two types of packages:

1.  Python wheels (.whl)
2.  SCIE (Self-Contained Installable Executables)

To build:

```bash
# Build wheels or SCIE packages
./scripts/build-wheel.sh
./scripts/build-scies.sh
```

All built packages will be placed in the `dist/` directory.

## License

Refer to the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE).