# Backend.AI: Containerized Computing Platform for AI and ML 

**Backend.AI is a powerful, container-based platform that simplifies and accelerates AI/ML workloads by providing a streamlined environment for diverse frameworks and hardware.** ([Go to the original repository](https://github.com/lablup/backend.ai))

## Key Features

*   **Containerized Environment:** Hosts popular computing and ML frameworks in isolated containers.
*   **Accelerator Support:** Offers pluggable support for various accelerators, including CUDA GPUs, ROCm GPUs, TPUs, and more.
*   **Multi-Tenant Computing:** Allocates and isolates resources for multi-tenant computing sessions on-demand or in batches.
*   **Flexible Job Scheduling:** Uses a customizable job scheduler ("Sokovan") for efficient resource management.
*   **API Access:** All functionalities are accessible through REST and GraphQL APIs.
*   **Client SDKs:** Provides Python, Java, JavaScript, and PHP SDKs for easy integration.

## Core Components

### Manager

*   **Description:** The control plane for the cluster, routing API requests and managing agents.
*   **Functionality:** Monitors and scales the cluster of agents.
*   **Plugins:** Interfaces for scheduler, agent selection, web applications, and monitoring.

### Agent

*   **Description:** Manages individual server instances and container lifecycles.
*   **Functionality:** Launches and destroys Docker containers where REPL daemons (kernels) run.
*   **Plugins:** Supports accelerator and monitoring plugins.

### Storage Proxy

*   **Description:** Provides a unified abstraction layer over network storage devices.
*   **Functionality:** Includes vendor-specific enhancements like performance metrics and operation acceleration.

### Webserver

*   **Description:** Hosts the web UI for end-users and basic administration.
*   **Functionality:** Serves a single-page application for web UI access.

### Kernels

*   **Description:** Compute environment recipes (Dockerfiles) to build container images.

### Jail

*   **Description:** A programmable sandbox for system call filtering in Rust.

### Hook

*   **Description:** A set of libc overrides for resource control and web-based interactive stdin.

### Client SDKs

*   Python (provides the command-line interface)
   * `pip install backend.ai-client`
   * [GitHub](https://github.com/lablup/backend.ai/tree/main/src/ai/backend/client)
*   Java
   * [GitHub](https://github.com/lablup/backend.ai-client-java)
*   Javascript
   * `npm install backend.ai-client`
   * [GitHub](https://github.com/lablup/backend.ai-client-js)
*   PHP (under preparation)
   * `composer require lablup/backend.ai-client`
   * [GitHub](https://github.com/lablup/backend.ai-client-php)

## Plugins

*   **Accelerator Plugins:**
    *   CUDA (`ai.backend.accelerator.cuda`)
    *   ROCm
*   **Monitoring Plugins:**
    *   Datadog (`ai.backend.monitor.stats`)
    *   Sentry (`ai.backend.monitor.error`)

## Getting Started

### Installation for Single-node Development

Run `scripts/install-dev.sh` after cloning this repository.

### Installation for Multi-node Tests &amp; Production

Please consult [our documentation](http://docs.backend.ai) for community-supported materials.

## Accessing Compute Sessions

*   **Jupyter:** Jupyter and JupyterLab support.
*   **Web-based terminal:** ttyd support.
*   **SSH:** SSH/SFTP/SCP support.
*   **VSCode:** web-based VSCode support.

## Working with Storage

Backend.AI provides vfolders (virtual folders) for cloud storage that can be mounted into any computation sessions.

## Building Packages

```bash
# Build wheels or SCIE packages
./scripts/build-wheels.sh
./scripts/build-scies.sh
```
## License

Refer to [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE).