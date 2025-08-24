# Backend.AI: Your All-in-One Platform for Containerized Computing and Machine Learning

**Backend.AI** is a powerful, container-based platform designed to simplify and accelerate your computing and machine-learning workflows. [Explore the original repository](https://github.com/lablup/backend.ai) to learn more.

[![PyPI release version](https://badge.fury.io/py/backend.ai-manager.svg)](https://pypi.org/project/backend.ai-manager/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/backend.ai-manager.svg)
![Wheels](https://img.shields.io/pypi/wheel/backend.ai-manager.svg)
[![Gitter](https://badges.gitter.im/lablup/backend.ai.svg)](https://gitter.im/lablup/backend.ai)

**Key Features:**

*   **Containerized Computing:** Effortlessly run popular computing and machine learning frameworks within isolated containers.
*   **Heterogeneous Accelerator Support:**  Leverage a wide array of accelerators, including CUDA GPUs, ROCm GPUs, TPUs, and IPUs.
*   **Multi-Tenant & Resource Management:** Efficiently allocate and isolate resources for multi-tenant computations with customizable job schedulers.
*   **REST & GraphQL APIs:**  Seamlessly integrate and interact with Backend.AI through its robust API endpoints.
*   **Flexible Access:** Access computation sessions via Jupyter, web-based terminals, SSH, and VS Code.
*   **Virtual Folder (vfolder) for Storage**: Provides an abstraction layer on top of existing network-based storages.

## Core Components

### Manager

*   The central control plane for managing the computing cluster, routing API requests, and monitoring agents.

### Agent

*   Manages individual server instances and launches/destroys Docker containers.

### Storage Proxy

*   Provides a unified abstraction over multiple different network storage devices.

### Webserver

*   Hosts the web UI for end-users and administration.

### Kernels

*   Computing environment recipes (Dockerfile) to build the container images.

## Additional Components

*   **Jail:** A programmable sandbox for security.
*   **Hook:** A set of libc overrides for resource control.
*   **Client SDKs:** Python, Java, Javascript, and PHP SDKs available for easy integration.

## Getting Started

### Single-Node Development

1.  Clone the repository.
2.  Run `scripts/install-dev.sh` to set up a development environment.

### Multi-Node Tests & Production

Consult the [official documentation](http://docs.backend.ai) for deployment instructions.

## Plugins

Extend the functionality of Backend.AI with plugins for:

*   **Accelerators:** CUDA, ROCm, and more.
*   **Monitoring:** Datadog, Sentry.

## Version Compatibility

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

## License

Backend.AI is licensed under LGPLv3 for server-side components and MIT for client SDKs and shared libraries.  See the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE) for details.