# Backend.AI: Your Container-Based Computing Platform for Modern Workloads

Backend.AI is a powerful platform designed to streamline and accelerate your computing and machine learning workflows, providing a flexible and scalable environment for diverse applications. [Explore the Backend.AI repository](https://github.com/lablup/backend.ai) for a comprehensive container-based solution.

## Key Features

*   **Container-Based Computing:** Leverages containerization for resource isolation and efficient execution of diverse workloads.
*   **Framework Support:** Hosts popular computing and machine learning frameworks, including support for CUDA GPUs, ROCm GPUs, and various NPUs.
*   **Multi-Language Support:** Enables the execution of code in a wide range of programming languages.
*   **Resource Management:** Allocates and isolates computing resources for multi-tenant computation sessions.
*   **API-Driven:** Offers REST and GraphQL APIs for seamless integration and automation.
*   **Flexible Scheduling:** Provides customizable job schedulers for on-demand or batch processing.
*   **Web-Based Access:** Provides Web UI via the web server.
*   **Client SDKs:** Client SDKs are available in Python, Java, Javascript, and PHP to ease integration.
*   **Extensible with Plugins:** Supports plugins for accelerators, monitoring, and other enhancements.

## Core Components

*   **Manager:** Manages the cluster, routes API requests, and scales resources.
*   **Agent:** Controls individual server instances and manages container lifecycles.
*   **Storage Proxy:** Provides a unified interface for various storage devices.
*   **Webserver:** Hosts the web UI for user access and administration.

## Getting Started

### Installation for Single-node Development

1.  Clone the repository.
2.  Run `./scripts/install-dev.sh` to set up a development environment. This requires `sudo` and a modern Linux or macOS system.

### Installation for Multi-node Tests &amp; Production

Consult the [Backend.AI documentation](http://docs.backend.ai) for detailed multi-node deployment instructions.

## Accessing Compute Sessions

Backend.AI provides several ways to access computation sessions (kernels):

*   **Jupyter:** Leverage Jupyter and JupyterLab within the containers.
*   **Web-based terminal:** Use the built-in ttyd support.
*   **SSH:** Secure SSH access with auto-generated key pairs.
*   **VSCode:** Use web-based VSCode support.

## Plugins

Extend Backend.AI's functionality using plugins, including:

*   `backendai_accelerator_v21`: Accelerator plugins for CUDA, ROCm, and more.
*   `backendai_monitor_stats_v10`: Statistics collection (e.g., Datadog).
*   `backendai_monitor_error_v10`: Exception tracking (e.g., Sentry).

## Version Compatibility

| Backend.AI Core Version | Python Version | Pantsbuild version |
|:-----------------------:|:--------------:|:------------------:|
| 25.06.x ~               | 3.13.x         | 2.23.x             |
| 24.03.x / 24.09.x ~ 25.05.x      | 3.12.x         | 2.21.x    |
| 23.03.x / 23.09.x       | 3.11.x         | 2.19.x             |
| 22.03.x / 22.09.x       | 3.10.x         |                    |
| 21.03.x / 21.09.x       | 3.8.x          |                    |

## Building Packages

Build Python wheels and SCIE (Self-Contained Installable Executables) packages:

```bash
./scripts/build-wheel.sh
./scripts/build-scies.sh
```

Built packages are located in the `dist/` directory.

## License

This project is licensed under the [LGPLv3](https://github.com/lablup/backend.ai/blob/main/LICENSE) license.