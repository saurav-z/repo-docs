# Backend.AI: The Containerized Computing Platform for Modern Workloads

[Backend.AI](https://github.com/lablup/backend.ai) is a powerful, container-based platform designed to streamline your computing and machine learning workflows.

## Key Features

*   **Containerized Execution:** Run computations in isolated, containerized environments for reproducibility and resource management.
*   **Multi-Framework Support:**  Seamlessly supports popular computing and ML frameworks and diverse programming languages.
*   **Heterogeneous Accelerator Support:**  Integrates with various accelerators, including CUDA GPUs, ROCm GPUs, and other NPUs, optimizing performance.
*   **On-Demand or Batch Processing:** Flexible resource allocation for both interactive sessions and batch jobs using the "Sokovan" orchestrator.
*   **API-Driven:** All functionalities are accessible through REST and GraphQL APIs for easy integration.
*   **Web-Based Access:** Provides web-based access to computing sessions for tools like Jupyter, web-based terminals, SSH, and VSCode.
*   **Storage Abstraction:** Offers a vfolder system on top of network storage for simplified data management.
*   **Comprehensive SDKs:** Includes Python, Java, JavaScript, and PHP client SDKs to integrate with your applications.

## Architecture

Backend.AI is built around several key components:

*   **Manager:** The central control plane that manages the cluster and routes API requests.
*   **Agent:** Deployed on each node, managing container execution and resources.
*   **Storage Proxy:** Abstracts storage operations across various network storage devices.
*   **Webserver:** Hosts the web UI for user access and administration.

## Getting Started

### Installation for Single-node Development

1.  Clone the repository.
2.  Run `./scripts/install-dev.sh` (requires `sudo` and a modern Python).

### Installation for Multi-node Tests &amp; Production

Refer to the [documentation](http://docs.backend.ai) for detailed instructions or contact contact@lablup.com for professional support.

## Core Components

*   **Manager:**
    *   [README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/manager/README.md)
*   **Agent:**
    *   [README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/agent/README.md)
*   **Storage Proxy:**
    *   [README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/storage/README.md)
*   **Webserver:**
    *   [README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/web/README.md)

## Contributing

Backend.AI welcomes contributions!  Please refer to the contribution guidelines and [license](https://github.com/lablup/backend.ai/blob/main/LICENSE) for details.

## License

Backend.AI is licensed under LGPLv3 for server-side components and MIT for other shared libraries and client SDKs.