# Backend.AI: Your Containerized Computing Platform for AI and More

**Backend.AI empowers you to run diverse computing frameworks and programming languages with ease and efficiency.** [Explore the source code](https://github.com/lablup/backend.ai).

[![PyPI release version](https://badge.fury.io/py/backend.ai-manager.svg)](https://pypi.org/project/backend.ai-manager/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/backend.ai-manager.svg)
![Wheels](https://img.shields.io/pypi/wheel/backend.ai-manager.svg)
[![Gitter](https://badges.gitter.im/lablup/backend.ai.svg)](https://gitter.im/lablup/backend.ai)

## Key Features

*   **Containerized Computing:** Leverage Docker containers for consistent and isolated computing environments.
*   **Heterogeneous Accelerator Support:** Seamlessly integrate and utilize CUDA GPU, ROCm GPU, and other NPUs for accelerated computing.
*   **Multi-Tenant Compute Sessions:** Allocate and isolate resources for on-demand or batch computations with custom job schedulers.
*   **REST and GraphQL APIs:** Access all functionality through robust and flexible APIs for easy integration.
*   **Flexible Compute Session Access:** Access sessions via Jupyter, web-based terminals (ttyd), SSH, and VSCode.
*   **Virtual Folder Storage:** Utilize vfolders to manage storage and share data across compute sessions with differentiated privileges.
*   **Client SDKs:** SDKs available in Python, Java, Javascript, and PHP for easy integration with your applications.

## Core Components

### Manager

*   **Description:** Orchestrates API requests, manages agents, and scales the cluster.
*   **Location:** `src/ai/backend/manager`
*   **Read More:** [Manager README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/manager/README.md)

### Agent

*   **Description:** Manages individual server instances and launches/destroys Docker containers.
*   **Location:** `src/ai/backend/agent`
*   **Read More:** [Agent README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/agent/README.md)

### Storage Proxy

*   **Description:** Provides a unified abstraction over multiple different network storage devices.
*   **Location:** `src/ai/backend/storage`
*   **Read More:** [Storage Proxy README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/storage/README.md)

### Webserver

*   **Description:** Hosts the web UI for end-users and basic administration.
*   **Location:** `src/ai/backend/web`
*   **Read More:** [Webserver README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/web/README.md)

## Getting Started

### Installation for Single-node Development

1.  Clone the repository.
2.  Run `scripts/install-dev.sh` (requires `sudo` and a modern Linux/macOS system with Python).

### Installation for Multi-node Tests &amp; Production

Consult the [Backend.AI documentation](http://docs.backend.ai) or contact the sales team for professional support.

## Plugins

Extend Backend.AI's functionality with plugins, including:

*   **Accelerator Plugins:** CUDA, ROCm, and more.
*   **Monitoring Plugins:** Datadog and Sentry integrations.

## Legacy Components

*   Media: Front-end support for multi-media outputs.
*   IDE and Editor Extensions: VS Code and Atom extensions (Jupyter Lab, VS Code Server, or SSH connections are now recommended).

## Python Version Compatibility

See the table in the original README.

## Building Packages

Build Python wheels and SCIE executables:

```bash
./scripts/build-wheels.sh
./scripts/build-scies.sh
```

Built packages are located in the `dist/` directory.

## License

Backend.AI is licensed under LGPLv3 for server-side components and MIT for shared libraries and client SDKs. See the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE).