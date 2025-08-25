# Backend.AI: A Container-Based Computing Cluster Platform for Modern AI & HPC

[![PyPI release version](https://badge.fury.io/py/backend.ai-manager.svg)](https://pypi.org/project/backend.ai-manager/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/backend.ai-manager.svg)
![Wheels](https://img.shields.io/pypi/wheel/backend.ai-manager.svg)
[![Gitter](https://badges.gitter.im/lablup/backend.ai.svg)](https://gitter.im/lablup/backend.ai)

Backend.AI is a cutting-edge platform that provides a streamlined, container-based environment for running a wide range of computing and Machine Learning workloads. [Check out the original repository](https://github.com/lablup/backend.ai) for more details.

## Key Features

*   **Containerized Workloads:** Run your applications within isolated containers for consistent and reproducible results.
*   **Framework Support:** Seamlessly hosts popular computing and ML frameworks, including TensorFlow, PyTorch, and more.
*   **Language Agnostic:** Supports diverse programming languages such as Python, R, and others.
*   **Accelerator Support:** Built-in support for heterogeneous accelerators, including CUDA GPUs, ROCm GPUs, and various NPUs (Rebellions, FuriosaAI, HyperAccel, Google TPU, Graphcore IPU, etc.).
*   **Resource Management:** Allocates and isolates computing resources for multi-tenant sessions with a customizable job scheduler, "Sokovan."
*   **API Access:** All functionalities are exposed through REST and GraphQL APIs, enabling easy integration and automation.
*   **Flexible Access:** Securely access compute sessions via web-based terminals, Jupyter notebooks, SSH, and VSCode.
*   **Storage Abstraction:** Provides a unified abstraction layer (vfolders) for network-based storage, simplifying data management.

## Core Components

### Manager

*   **Description:** The central control plane, routing API requests and managing the cluster of agents.
*   **Key Functions:** Cluster monitoring, scaling, and API request routing.

### Agent

*   **Description:** Manages individual server instances and container lifecycles.
*   **Key Functions:** Container launch/destruction, resource monitoring, and self-registration.

### Storage Proxy

*   **Description:** Provides a unified interface for various network storage devices.
*   **Key Functions:** Simplifies data access with performance metrics and API acceleration.

### Webserver

*   **Description:** Hosts the web UI for end-users and basic administration.
*   **Key Functions:** Serves the Single Page Application (SPA) web interface.

## Other Components

*   **Kernels:** Pre-built computing environment recipes (Dockerfiles).
*   **Jail:** ptrace-based system call filtering for security.
*   **Hook:** Custom libc overrides for resource control.
*   **Client SDKs:** Python, Java, JavaScript, and PHP libraries for easy integration.

## Getting Started

### Installation for Single-node Development

Run `./scripts/install-dev.sh` after cloning this repository. This script installs dependencies, including Docker, and sets up the development environment. Requires `sudo` and a modern Python version on Linux or macOS.

### Installation for Multi-node Tests & Production

Refer to the [official documentation](http://docs.backend.ai) for detailed instructions. For commercial support, contact contact@lablup.com.

## Building Packages

Build Python wheels and SCIE (Self-Contained Installable Executables) using the provided scripts:

```bash
./scripts/build-wheels.sh
./scripts/build-scies.sh
```

Built packages will be located in the `dist/` directory.

## License

Backend.AI server-side components are licensed under LGPLv3. Shared libraries and client SDKs are distributed under the MIT license. See the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE) for details.