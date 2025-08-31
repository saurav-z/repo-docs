# Backend.AI: Your All-in-One Containerized Computing Platform

**Backend.AI** is a powerful platform that simplifies and streamlines container-based computing, making it easier to run your favorite machine learning frameworks and programming languages.  Explore its capabilities at the original repository: [https://github.com/lablup/backend.ai](https://github.com/lablup/backend.ai).

[![PyPI release version](https://badge.fury.io/py/backend.ai-manager.svg)](https://pypi.org/project/backend.ai-manager/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/backend.ai-manager.svg)
![Wheels](https://img.shields.io/pypi/wheel/backend.ai-manager.svg)
[![Gitter](https://badges.gitter.im/lablup/backend.ai.svg)](https://gitter.im/lablup/backend.ai)

## Key Features

*   **Container-Based Computing:** Leverage the power of containers for isolated and reproducible computing environments.
*   **Framework & Language Support:**  Easily host popular computing and ML frameworks along with a wide variety of programming languages.
*   **Heterogeneous Accelerator Support:**  Includes support for CUDA GPU, ROCm GPU, Rebellions, FuriosaAI, HyperAccel, Google TPU, Graphcore IPU, and other NPUs.
*   **Resource Management:** Efficiently allocates and isolates resources for multi-tenant computation sessions, either on-demand or in batches.
*   **API-Driven:** All functionalities are exposed via REST and GraphQL APIs for seamless integration.
*   **Flexible Job Scheduling:**  Customizable job schedulers are available, managed by the Sokovan orchestrator.
*   **Secure Access:**  Provides secure access to computation sessions through web sockets and supports tools like Jupyter, web-based terminals, SSH, and VSCode.
*   **Storage Abstraction:** Simplifies storage management through vfolders, providing a cloud-like storage experience.

## Core Components

### Manager

*   **Description:** Routes API requests and manages the cluster of agents.
*   **Repository:** `src/ai/backend/manager`
*   **Plugins:**  Supports various plugin interfaces for customization.

### Agent

*   **Description:**  Manages individual server instances and container lifecycles.
*   **Repository:** `src/ai/backend/agent`
*   **Plugins:** Supports accelerator and monitoring plugins.

### Storage Proxy

*   **Description:** Provides a unified abstraction for various network storage devices.
*   **Repository:** `src/ai/backend/storage`

### Webserver

*   **Description:** Hosts the web UI for user access and administration.
*   **Repository:** `src/ai/backend/web`

### Kernels

*   **Description:** Contains computing environment recipes (Dockerfiles) to build container images.
*   **Repository:** [https://github.com/lablup/backend.ai-kernels](https://github.com/lablup/backend.ai-kernels)

### Jail

*   **Description:**  A ptrace-based system call filtering sandbox written in Rust.
*   **Repository:** [https://github.com/lablup/backend.ai-jail](https://github.com/lablup/backend.ai-jail)

### Hook

*   **Description:**  Provides libc overrides for resource control and web-based stdin.
*   **Repository:** [https://github.com/lablup/backend.ai-hook](https://github.com/lablup/backend.ai-hook)

## Client SDKs

*   **Python:** `pip install backend.ai-client` - [Source](https://github.com/lablup/backend.ai/tree/main/src/ai/backend/client)
*   **Java:** [GitHub Releases](https://github.com/lablup/backend.ai-client-java)
*   **Javascript:** `npm install backend.ai-client` - [Source](https://github.com/lablup/backend.ai-client-js)
*   **PHP:** `composer require lablup/backend.ai-client` - [Source](https://github.com/lablup/backend.ai-client-php)

## Plugins

Backend.AI leverages plugins via Python package entrypoints, extending functionality.

*   `backendai_accelerator_v21`:  Accelerator plugins including CUDA and ROCm.
*   `backendai_monitor_stats_v10`: Statistics collectors based on the Datadog API.
*   `backendai_monitor_error_v10`: Exception collectors based on the Sentry API.

## Getting Started

### Installation for Single-node Development

Run `scripts/install-dev.sh` after cloning this repository.

### Installation for Multi-node Tests &amp; Production

Consult [our documentation](http://docs.backend.ai). Contact contact@lablup.com for professional support.

## Building Packages

The project supports building two types of packages:

1.  Python wheels (.whl)
2.  SCIE (Self-Contained Installable Executables)

To build:

```bash
# Build wheels or SCIE packages
./scripts/build-wheels.sh
./scripts/build-scies.sh
```

## Python Version Compatibility

| Backend.AI Core Version | Python Version | Pantsbuild version |
|:-----------------------:|:--------------:|:------------------:|
| 25.06.x ~               | 3.13.x         | 2.23.x             |
| 24.03.x / 24.09.x ~ 25.05.x      | 3.12.x         | 2.21.x    |
| 23.03.x / 23.09.x       | 3.11.x         | 2.19.x             |
| 22.03.x / 22.09.x       | 3.10.x         |                    |
| 21.03.x / 21.09.x       | 3.8.x          |                    |

## License

Refer to the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE).