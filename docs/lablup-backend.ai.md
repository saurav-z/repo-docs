# Backend.AI: Unleash the Power of Containerized Computing

Backend.AI is a versatile platform that simplifies the deployment and management of computing clusters, offering a streamlined environment for your machine learning and data science workloads. **[Get started with Backend.AI on GitHub](https://github.com/lablup/backend.ai)**

## Key Features

*   **Containerized Computing:** Leverages Docker containers for reproducible and isolated computing environments.
*   **Broad Framework Support:** Hosts popular computing and ML frameworks, including TensorFlow, PyTorch, and more.
*   **Multi-Language Support:** Supports diverse programming languages such as Python, R, and Julia.
*   **Heterogeneous Accelerator Support:** Integrates with various accelerators like CUDA GPUs, ROCm GPUs, and NPUs, enabling accelerated computing.
*   **Resource Management:** Allocates and isolates computing resources for multi-tenant computation sessions.
*   **REST & GraphQL APIs:** Provides comprehensive APIs for seamless integration and automation.
*   **Job Scheduling:** Offers customizable job schedulers with its orchestrator "Sokovan."
*   **Secure Access:** Provides secure access to computation sessions through Jupyter, web-based terminals, SSH, and VSCode.
*   **Flexible Storage:** Supports virtual folders (vfolders) for seamless integration with various storage solutions (e.g., NFS/SMB).
*   **Client SDKs:** Easily integrate with backend.ai via Python, Java, Javascript, and PHP (under preparation) SDKs.
*   **Extensible with Plugins:** Supports plugins for accelerators, monitoring, and other custom functionalities.

## Core Components

*   **Manager:** The central control plane that manages cluster resources, schedules jobs, and routes API requests.
*   **Agent:** Runs on each compute node, manages container lifecycles, and interacts with the Manager.
*   **Storage Proxy:** Abstracts and manages storage operations, providing a unified interface to various storage backends.
*   **Webserver:** Hosts the web UI for user interaction and administration.
*   **Kernels:** Computing environment recipes to build container images.
*   **Jail:** A programmable sandbox for security.
*   **Hook:** A set of libc overrides for resource control.
*   **Client SDKs:** Libraries for Python, Java, Javascript, and PHP (under preparation).

## Getting Started

### Installation for Single-node Development

Run `scripts/install-dev.sh` after cloning this repository. This script sets up a development environment with required dependencies, including Docker.

### Installation for Multi-node Tests & Production

Consult [our documentation](http://docs.backend.ai) for community-supported materials. Contact the sales team (contact@lablup.com) for professional paid support and deployment options.

## Plugin Ecosystem

*   `backendai_accelerator_v21`: Accelerator plugins (CUDA, ROCm, and more).
*   `backendai_monitor_stats_v10`: Statistics collection via Datadog API.
*   `backendai_monitor_error_v10`: Exception collection via Sentry API.

## Version Compatibility

Provides a table that shows compatible versions of Backend.AI, Python, and Pantsbuild.

## Build Packages

Build Python wheels (.whl) and SCIE (Self-Contained Installable Executables) packages using the scripts.

## License

Refer to the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE) for licensing details.