# Backend.AI: Container-Based Computing for Modern Workloads

**Backend.AI is a powerful platform designed to streamline and accelerate your compute-intensive tasks by leveraging containerization and diverse hardware acceleration.**  [Explore the original repository](https://github.com/lablup/backend.ai)

## Key Features

*   **Containerized Computing:** Run your workloads in isolated, containerized environments for consistent and reproducible results.
*   **Framework & Language Support:** Hosts popular computing/ML frameworks and diverse programming languages.
*   **Heterogeneous Accelerator Support:**  Integrates with CUDA GPU, ROCm GPU, Rebellions, FuriosaAI, HyperAccel, Google TPU, Graphcore IPU and other NPUs for optimized performance.
*   **Resource Management:** Allocates and isolates computing resources for multi-tenant sessions, on-demand or in batches.
*   **Flexible Job Scheduling:** Customizable job schedulers with "Sokovan" orchestrator.
*   **API Access:** Exposes all functions through REST and GraphQL APIs, providing flexible integration options.
*   **Client SDKs:** Comprehensive SDKs in Python, Java, JavaScript, and PHP.
*   **Secure Access:** Provides secure websocket tunneling into compute sessions with Jupyter, web-based terminal, SSH, and VSCode support.
*   **Storage Abstraction:** Offers a vfolder abstraction layer over network storage for easy data management.
*   **Extensible:**  Supports plugins for accelerators, monitoring, and more.

## Core Components

*   **Manager:** The cluster control-plane, routing API requests and managing cluster resources.
*   **Agent:**  Per-node controller managing instances and launching/destroying containers.
*   **Storage Proxy:** Provides a unified interface for various network storage devices.
*   **Webserver:**  Hosts the web UI for end-users and basic administration.
*   **Kernels:**  Container images providing computing environment recipes.
*   **Jail:** ptrace-based system call filtering for secure sandboxing.
*   **Hook:**  Set of libc overrides for resource control.

## Getting Started

*   **Single-node Development:**  Use the `scripts/install-dev.sh` script after cloning the repository.
*   **Multi-node Tests & Production:** Consult the documentation ([http://docs.backend.ai](http://docs.backend.ai))

## Licensing

Backend.AI server-side components are licensed under LGPLv3, and the client SDKs are distributed under the MIT license.  For commercial consulting and licensing inquiries, please contact contact-at-lablup-com.