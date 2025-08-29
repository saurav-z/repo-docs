# Backend.AI: Your All-in-One Platform for Containerized Computing and Machine Learning

Backend.AI is a powerful, open-source platform for running and managing containerized computing and machine learning workloads, offering flexible resource allocation and diverse framework support. Access the original repo [here](https://github.com/lablup/backend.ai).

## Key Features

*   **Container-Based Computing:** Leverages containerization for isolated and reproducible computing environments.
*   **Framework and Language Support:** Hosts popular computing and ML frameworks, including CUDA GPU, ROCm GPU, Rebellions, FuriosaAI, HyperAccel, Google TPU, Graphcore IPU and other NPUs. and diverse programming languages.
*   **Heterogeneous Accelerator Support:** Provides pluggable support for various accelerators, including GPUs and TPUs.
*   **Resource Management:** Allocates and isolates computing resources for multi-tenant sessions, both on-demand and in batches.
*   **API-Driven:** Exposes all functions via REST and GraphQL APIs for easy integration.
*   **Flexible Job Scheduling:** Offers customizable job schedulers with the "Sokovan" orchestrator.
*   **Multi-Node Support:** Designed for cluster deployments.
*   **Client SDKs:** Offers client SDKs in Python, Java, Javascript, and PHP for easy integration.
*   **Web-Based Access:** Provides web UI access to Jupyter, web-based terminal, SSH, and VSCode.
*   **Storage Abstraction:** Offers vfolders (virtual folders) for easy sharing and access to storage across computing sessions.

## Core Components

### Manager

*   **Description:** The control plane that routes external API requests and manages the cluster.
*   **Functionality:** Monitors and scales the cluster, routes API requests to agents.
*   **Repository:** `src/ai/backend/manager`

### Agent

*   **Description:** Manages individual server instances and launches/destroys containers.
*   **Functionality:** Launches and manages Docker containers where kernels run, self-registers to the instance registry.
*   **Repository:** `src/ai/backend/agent`

### Storage Proxy

*   **Description:** Provides a unified abstraction over network storage.
*   **Functionality:** Simplifies access to network storage with performance metrics.
*   **Repository:** `src/ai/backend/storage`

### Webserver

*   **Description:** Hosts the web UI for end-users and administrators.
*   **Functionality:** Provides access to the web UI.
*   **Repository:** `src/ai/backend/web`

## Getting Started

### Installation for Single-Node Development

Run `scripts/install-dev.sh` after cloning the repository. This script sets up a development environment.

### Installation for Multi-Node Tests & Production

Please consult the [official documentation](http://docs.backend.ai). Contact the sales team (contact@lablup.com) for professional paid support.

### Building Packages

Backend.AI supports building Python wheels and SCIE (Self-Contained Installable Executables).

```bash
# Build wheels or SCIE packages
./scripts/build-wheels.sh
./scripts/build-scies.sh
```

Built packages are located in the `dist/` directory.

## Accessing Compute Sessions

Backend.AI provides websocket tunneling into individual computation sessions (containers),
so that users can use their browsers and client CLI to access in-container applications directly
in a secure way.

* Jupyter: data scientists' favorite tool
* Web-based terminal
* SSH
* VSCode

## Plugins

Backend.AI supports plugins via Python package entrypoints. Key entrypoints include:

*   `backendai_accelerator_v21`: Accelerator plugins (CUDA, ROCm, etc.)
*   `backendai_monitor_stats_v10`: Statistics collectors (e.g., Datadog)
*   `backendai_monitor_error_v10`: Exception collectors (e.g., Sentry)

## Licensing

Server-side components are licensed under LGPLv3. Shared libraries and client SDKs are distributed under the MIT license. See the [LICENSE](https://github.com/lablup/backend.ai/blob/main/LICENSE) file.