# Backend.AI: Your All-in-One Containerized Computing Platform

[Backend.AI](https://github.com/lablup/backend.ai) is a powerful, open-source platform that streamlines your computing and machine learning workloads by leveraging containerization and supporting a wide range of accelerators.

[![PyPI release version](https://badge.fury.io/py/backend.ai-manager.svg)](https://pypi.org/project/backend.ai-manager/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/backend.ai-manager.svg)
![Wheels](https://img.shields.io/pypi/wheel/backend.ai-manager.svg)
[![Gitter](https://badges.gitter.im/lablup/backend.ai.svg)](https://gitter.im/lablup/backend.ai)

## Key Features

*   **Containerized Computing:** Run your applications and frameworks within isolated containers for consistent and reproducible environments.
*   **Heterogeneous Accelerator Support:** Seamlessly utilize a variety of accelerators, including CUDA GPUs, ROCm GPUs, and more.
*   **Multi-Tenant Environment:**  Safely and efficiently share computing resources among multiple users with resource allocation and isolation.
*   **REST & GraphQL APIs:**  Access all functionalities through well-defined APIs for easy integration and automation.
*   **Flexible Job Scheduling:**  Customize job scheduling to optimize resource utilization and meet your specific needs.
*   **Integrated Storage Abstraction:** Simplified management of network-based storage with vfolders.
*   **Client SDKs:** Convenient SDKs available in Python, Java, JavaScript, and PHP to simplify integration.
*   **Web-Based Access:** Access compute sessions via web-based terminal, JupyterLab, VSCode, and more.

## Core Components

*   **Manager:** The central control plane for managing the cluster, handling API requests, and scaling agents.
*   **Agent:** Deploys and manages individual server instances, launching and destroying Docker containers.
*   **Storage Proxy:**  Provides a unified abstraction over multiple network storage devices.
*   **Webserver:** Hosts the web UI for end-users and basic administration.
*   **Kernels:** Computing environment recipes (Dockerfiles) to build the container images to execute.
*   **Jail:** A programmable sandbox implemented using ptrace-based system call filtering written in Rust.
*   **Hook:** A set of libc overrides for resource control and web-based interactive stdin (paired with agents).

## Plugins

Extend Backend.AI's capabilities with plugins. Available plugins include:

*   **Accelerator Plugins:** CUDA, ROCm, and more.
*   **Monitoring Plugins:** Datadog, Sentry, and more.

## Getting Started

### Installation for Single-node Development

1.  Clone the repository:
```bash
git clone https://github.com/lablup/backend.ai
cd backend.ai
```

2.  Run the development setup script:
```bash
./scripts/install-dev.sh
```

### Installation for Multi-node Tests & Production

Please consult [our documentation](http://docs.backend.ai) or contact the sales team (contact@lablup.com) for more details.

## Building Packages

Backend.AI supports building Python wheels and SCIE (Self-Contained Installable Executables):

```bash
./scripts/build-wheels.sh
./scripts/build-scies.sh
```

Built packages will be placed in the `dist/` directory.

## License

Backend.AI is licensed under the LGPLv3 for server-side components, while other shared libraries and client SDKs are distributed under the MIT license.  See the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE) for details.