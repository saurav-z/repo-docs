# Backend.AI: Unleash the Power of Containerized Computing and ML Frameworks

Backend.AI is a powerful platform that simplifies and streamlines access to computing resources, offering a seamless environment for data scientists and developers to run complex workloads. ([Original Repo](https://github.com/lablup/backend.ai))

## Key Features

*   **Container-Based Computing:** Hosts popular computing/ML frameworks and diverse programming languages within isolated containers for efficient resource utilization.
*   **Heterogeneous Accelerator Support:** Pluggable support for CUDA GPUs, ROCm GPUs, and various NPUs including FuriosaAI, HyperAccel, Google TPU, and Graphcore IPU.
*   **Multi-Tenant and Resource Management:** Allocates and isolates computing resources for multi-tenant computation sessions on-demand or in batches with customizable job schedulers via its own orchestrator named "Sokovan".
*   **REST and GraphQL APIs:**  All functionalities are exposed through robust REST and GraphQL APIs for seamless integration and automation.
*   **Flexible Access to Compute Sessions:** Provides secure access to compute sessions via Jupyter, web-based terminals (ttyd), SSH, and VSCode.
*   **Virtual Folder Storage:** Offers an abstraction layer over network storage (NFS/SMB) with vfolders for easy data management and sharing.
*   **Client SDKs:** Available SDKs in Python, Java, JavaScript, and PHP to simplify integration into existing workflows.

## Core Components

*   **Manager:** The central control plane for the cluster, handling API requests, agent management, and cluster scaling.
*   **Agent:** Manages individual server instances and launches/destroys Docker containers where REPL daemons (kernels) run.
*   **Storage Proxy:** Provides a unified interface for network storage devices, with performance enhancements.
*   **Webserver:** Hosts the web UI for end-users and basic administration.
*   **Kernels:** Container images (Dockerfiles) for building compute environments.
*   **Jail:**  A Rust-based system call filtering sandbox.
*   **Hook:**  A set of libc overrides for resource control and web-based interactive stdin.
*   **Client SDK Libraries:**  Python, Java, JavaScript, and PHP SDKs for easy integration.

## Getting Started

*   **Single-node Development:** Run `scripts/install-dev.sh` to quickly set up a development environment.
*   **Multi-node Tests & Production:** Refer to the official documentation for detailed deployment instructions.

## Plugins

Extend the functionality of Backend.AI with plugins through Python package entrypoints, including:

*   **Accelerators:** CUDA, ROCm, and more.
*   **Monitors:** Datadog and Sentry integration.

## Build and Package

*   Python wheels and SCIE packages are supported.
*   Use `./scripts/build-wheels.sh` and `./scripts/build-scies.sh` to build.
*   Packages are placed in the `dist/` directory.

## License

Backend.AI server-side components are licensed under LGPLv3, while shared libraries and client SDKs are distributed under the MIT license. See the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE) for details.