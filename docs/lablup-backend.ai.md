# Backend.AI: The Containerized Computing Platform for Modern Workloads

Backend.AI is a powerful, container-based computing platform that simplifies the deployment and management of diverse computing and machine learning workloads.  [Explore the original repository](https://github.com/lablup/backend.ai).

[![PyPI release version](https://badge.fury.io/py/backend.ai-manager.svg)](https://pypi.org/project/backend.ai-manager/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/backend.ai-manager.svg)
![Wheels](https://img.shields.io/pypi/wheel/backend.ai-manager.svg)
[![Gitter](https://badges.gitter.im/lablup/backend.ai.svg)](https://gitter.im/lablup/backend.ai)

## Key Features

*   **Containerized Computing:**  Leverage Docker containers for consistent and reproducible environments.
*   **Diverse Framework Support:** Hosts popular computing/ML frameworks and programming languages.
*   **Heterogeneous Accelerator Support:** Seamlessly integrates with CUDA GPUs, ROCm GPUs, TPUs, IPUs, and more.
*   **Resource Management:**  Efficiently allocates and isolates computing resources for multi-tenant sessions.
*   **REST & GraphQL APIs:**  Provides flexible and scalable API access to all functionalities.
*   **Customizable Scheduling:**  Offers job scheduling with its own orchestrator, "Sokovan."
*   **Web and CLI Access:** Access sessions through Jupyter, web-based terminals, SSH, and VSCode, enhancing user accessibility.
*   **Storage Abstraction:** Provides vfolders for cloud-based storage solutions.

## Core Components

### Manager

The central control plane for the cluster, handling API requests, monitoring, and scaling.

### Agent

Manages individual server instances, launching and destroying containerized kernels.

### Storage Proxy

Provides a unified abstraction over network storage devices.

### Webserver

Hosts the web UI for end-users and administration.

### Kernels

Computing environment recipes (Dockerfiles) that build container images.

### Jail

A programmable sandbox based on `ptrace`.

### Hook

A set of libc overrides for resource control and web-based interactive stdin.

## Client SDKs

Easily integrate Backend.AI with your applications using our client SDKs:

*   **Python:** `pip install backend.ai-client`
*   **Java:** Available via GitHub releases
*   **Javascript:** `npm install backend.ai-client`
*   **PHP:** `composer require lablup/backend.ai-client` (in preparation)

## Getting Started

*   **Single-node Development:** Run `./scripts/install-dev.sh` after cloning the repository.
*   **Multi-node Tests & Production:** Refer to the official documentation.

## Plugins

Backend.AI supports plugins for extensibility:

*   `backendai_accelerator_v21`: Accelerator plugins (CUDA, ROCm, etc.)
*   `backendai_monitor_stats_v10`: Statistics collectors (e.g., Datadog)
*   `backendai_monitor_error_v10`: Exception collectors (e.g., Sentry)

## Build Packages

*  Build wheels with `./scripts/build-wheel.sh`.
*  Build SCIE packages with `./scripts/build-scies.sh`.

## License

Backend.AI is licensed under LGPLv3 for server-side components and MIT for shared libraries and client SDKs.  See the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE) for details.