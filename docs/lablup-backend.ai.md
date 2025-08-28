# Backend.AI: The Open-Source Platform for Containerized Computing and AI

**Backend.AI empowers you to effortlessly run and manage containerized computing environments for diverse workloads, from machine learning to scientific simulations.** ([Original Repository](https://github.com/lablup/backend.ai))

## Key Features

*   **Container-Based Computing:**  Leverage Docker containers for consistent and reproducible environments.
*   **Multi-Framework Support:**  Seamlessly run popular frameworks like TensorFlow, PyTorch, and others.
*   **Heterogeneous Accelerator Support:** Includes support for CUDA GPUs, ROCm GPUs, and more.
*   **Resource Management:**  Efficiently allocate and isolate resources with a built-in scheduler.
*   **REST and GraphQL APIs:**  Integrate and control the platform via robust and modern APIs.
*   **Flexible Job Scheduling:** Customize job scheduling to meet your specific needs.
*   **Web-Based Access:** Easily access your compute sessions via Jupyter, web terminals, and more.
*   **Storage Abstraction:**  Utilize virtual folders (vfolders) for simplified data management.

## Core Components

*   **Manager:** The cluster control plane, managing agents and routing API requests.
*   **Agent:**  Manages individual server instances and launches containers.
*   **Storage Proxy:** Provides a unified interface to various network storage solutions.
*   **Webserver:** Hosts the web UI for user access and administration.
*   **Kernels:** Computing environment recipes (Dockerfiles) to build container images.
*   **Client SDKs:**  SDKs available in Python, Java, JavaScript, and PHP.
*   **Jail**: A programmable sandbox for system call filtering written in Rust.
*   **Hook**: A set of libc overrides for resource control and web-based interactive stdin

## Getting Started

### Installation for Single-node Development

1.  Clone this repository.
2.  Run `./scripts/install-dev.sh` (requires `sudo` and a modern Python).

### Installation for Multi-node Tests & Production

Consult the [Backend.AI documentation](http://docs.backend.ai) for detailed guidance.

### Accessing Compute Sessions

Backend.AI offers various ways to access your compute sessions:

*   Jupyter Notebooks and JupyterLab.
*   Web-based terminal (ttyd).
*   SSH access with auto-generated keypairs (compatible with IDEs).
*   Web-based VSCode support.

## Plugins

Backend.AI supports plugins for enhanced functionality:

*   `backendai_accelerator_v21`: Accelerator plugins (CUDA, ROCm, etc.)
*   `backendai_monitor_stats_v10`: Statistics collector plugins (e.g., Datadog).
*   `backendai_monitor_error_v10`: Exception collector plugins (e.g., Sentry).

## Version Compatibility

The Python version compatibility is detailed below.

| Backend.AI Core Version | Python Version | Pantsbuild version |
|:-----------------------:|:--------------:|:------------------:|
| 25.06.x ~               | 3.13.x         | 2.23.x             |
| 24.03.x / 24.09.x ~ 25.05.x      | 3.12.x         | 2.21.x    |
| 23.03.x / 23.09.x       | 3.11.x         | 2.19.x             |
| 22.03.x / 22.09.x       | 3.10.x         |                    |
| 21.03.x / 21.09.x       | 3.8.x          |                    |

## Building Packages

Backend.AI offers two types of packages:

1.  Python wheels (`.whl`)
2.  SCIE (Self-Contained Installable Executables)

To build packages, use:

```bash
./scripts/build-wheels.sh  # Build Python wheels
./scripts/build-scies.sh   # Build SCIE packages
```

Built packages are located in the `dist/` directory.

## License

Backend.AI is licensed under the LGPLv3 license for server-side components and MIT for client SDKs and shared libraries. See the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE) for details.