# Backend.AI: Your All-in-One Platform for Containerized Computing

Backend.AI is a powerful, open-source platform that streamlines and simplifies the deployment and management of computing clusters for diverse workloads.  [Visit the original repository](https://github.com/lablup/backend.ai) to get started!

## Key Features

*   **Container-Based Computing:**  Leverages containers for consistent and isolated execution environments.
*   **Framework & Language Support:** Hosts popular computing/ML frameworks and diverse programming languages.
*   **Heterogeneous Accelerator Support:**  Offers pluggable support for CUDA GPUs, ROCm GPUs, and other NPUs.
*   **Resource Management:** Allocates and isolates computing resources for multi-tenant sessions on-demand or in batches.
*   **Flexible Job Scheduling:**  Customizable job schedulers with the "Sokovan" orchestrator.
*   **API Access:**  All functionalities are exposed through REST and GraphQL APIs.
*   **Web-Based Access:**  Access your compute sessions through a web-based terminal, Jupyter, and more.
*   **Storage Abstraction:** Provides virtual folders (vfolders) for simplified storage management.
*   **Client SDKs:**  Python, Java, JavaScript, and PHP SDKs for easy integration.

## Core Components

### Manager

Manages the cluster, routes API requests, and monitors/scales the agents.

*   [Manager README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/manager/README.md)
*   **Plugins:** `backendai_scheduler_v10`, `backendai_agentselector_v10`, `backendai_hook_v20`, `backendai_webapp_v20`, `backendai_monitor_stats_v10`, `backendai_monitor_error_v10`

### Agent

Manages individual server instances and launches/destroys Docker containers.

*   [Agent README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/agent/README.md)
*   **Plugins:** `backendai_accelerator_v21`, `backendai_monitor_stats_v10`, `backendai_monitor_error_v10`

### Storage Proxy

Provides a unified abstraction over network storage devices.

*   [Storage Proxy README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/storage/README.md)

### Webserver

Hosts the web UI for end-users and basic administration tasks.

*   [Webserver README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/web/README.md)

### Kernels

Computing environment recipes (Dockerfiles) for building container images.

*   [Backend.AI Kernels](https://github.com/lablup/backend.ai-kernels)

### Jail

A programmable sandbox for enhanced security.

*   [Backend.AI Jail](https://github.com/lablup/backend.ai-jail)

### Hook

A set of libc overrides for resource control.

*   [Backend.AI Hook](https://github.com/lablup/backend.ai-hook)

### Client SDK Libraries

SDKs available for Python, Java, JavaScript, and PHP.

*   **Python:** `pip install backend.ai-client` ( [Client Source](https://github.com/lablup/backend.ai/tree/main/src/ai/backend/client))
*   **Java:**  ( [Java Client Source](https://github.com/lablup/backend.ai-client-java))
*   **JavaScript:** `npm install backend.ai-client` ( [JavaScript Client Source](https://github.com/lablup/backend.ai-client-js))
*   **PHP:**  `composer require lablup/backend.ai-client` ( [PHP Client Source](https://github.com/lablup/backend.ai-client-php))

## Plugins

Extend Backend.AI's functionality with plugins.

*   **Accelerator Plugins:** `backendai_accelerator_v21` (CUDA, ROCm, and more)
*   **Monitoring Plugins:** `backendai_monitor_stats_v10` (Datadog), `backendai_monitor_error_v10` (Sentry)

## Getting Started

### Installation for Single-Node Development

Run `scripts/install-dev.sh` after cloning this repository.

### Installation for Multi-Node Tests &amp; Production

See [our documentation](http://docs.backend.ai).

### Accessing Compute Sessions (Kernels)

*   Jupyter
*   Web-based terminal
*   SSH
*   VSCode

## Building Packages

Build Python wheels and SCIE (Self-Contained Installable Executables).

*   ```bash
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