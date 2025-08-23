# Backend.AI: Your Containerized Computing Cluster Platform

**Backend.AI empowers researchers and developers to easily run and scale their computing/ML workloads on a variety of hardware.** Learn more at the [Backend.AI GitHub repository](https://github.com/lablup/backend.ai).

## Key Features

*   **Container-Based:** Leverage Docker containers for consistent and reproducible environments.
*   **Framework Support:** Hosts popular computing and ML frameworks, including TensorFlow, PyTorch, and more.
*   **Multi-Language Support:** Supports diverse programming languages for flexible development.
*   **Heterogeneous Accelerator Support:** Integrates with CUDA GPUs, ROCm GPUs, and other NPUs (Rebellions, FuriosaAI, HyperAccel, Google TPU, Graphcore IPU).
*   **Resource Management:** Allocates and isolates computing resources for multi-tenant sessions, on-demand or in batches.
*   **API-Driven:** All functionality is exposed via REST and GraphQL APIs for easy integration.
*   **Job Scheduling:** Customizable job schedulers with the "Sokovan" orchestrator.
*   **Web-Based Access:** Provides Jupyter, web-based terminal, SSH, and VSCode support for easy access to compute sessions.
*   **Storage Abstraction:** Offers vfolders (virtual folders) for managing and sharing storage.

## Core Components

### Manager

The central control plane for managing the computing cluster.

*   Routes API requests.
*   Monitors and scales the cluster.

### Agent

Manages individual server instances and containers.

*   Launches and destroys Docker containers.

### Storage Proxy

Provides a unified abstraction over network storage devices.

### Webserver

Hosts the Backend.AI Web UI.

## Getting Started

### Installation for Single-node Development

Run `scripts/install-dev.sh` after cloning this repository. Requires `sudo` and a modern Python installed on Linux or macOS.

### Installation for Multi-node Tests & Production

Please consult [our documentation](http://docs.backend.ai).

### Accessing Compute Sessions (Kernels)

Backend.AI provides various ways to access your compute sessions.

*   **Jupyter:** Data science's favorite tool.
*   **Web-based terminal:** All container sessions have intrinsic ttyd support.
*   **SSH:** All container sessions have intrinsic SSH/SFTP/SCP support with auto-generated per-user SSH keypair.
*   **VSCode:** Most container sessions have intrinsic web-based VSCode support.

## Client SDKs

Easily integrate with Backend.AI using our client SDKs, available under the MIT License:

*   **Python:** `pip install backend.ai-client`
*   **Java**
*   **Javascript:** `npm install backend.ai-client`
*   **PHP**

## Plugins

Extend Backend.AI with plugins for accelerators and monitoring.

## Python Version Compatibility

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

Backend.AI server-side components are licensed under LGPLv3. Client SDKs are distributed under the MIT license.  See the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE).