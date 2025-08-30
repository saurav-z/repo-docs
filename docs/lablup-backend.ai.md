# Backend.AI: Your Containerized Computing and AI Platform

Backend.AI is a powerful, open-source platform that simplifies and streamlines your computing and machine learning workflows. Access the original repository [here](https://github.com/lablup/backend.ai).

## Key Features

*   **Container-Based Computing:** Leverage containerization for consistent and reproducible environments.
*   **Multi-Framework Support:** Hosts popular computing/ML frameworks and diverse programming languages.
*   **Heterogeneous Accelerator Support:** Pluggable support for various accelerators, including CUDA GPUs, ROCm GPUs, TPUs, and more.
*   **Resource Management:** On-demand or batch computation sessions with customizable job schedulers.
*   **REST and GraphQL APIs:** All functionalities are exposed through easy-to-use APIs.
*   **Secure Access:** Provides websocket tunneling for secure access to computation sessions through Jupyter, web-based terminals, SSH, and VSCode.
*   **Vfolder Storage:** Abstraction layer for managing network storage devices.
*   **Client SDKs:** Available SDKs in Python, Java, JavaScript, and PHP (under preparation) to easily integrate with your projects.

## Core Components

*   **Manager:** The cluster control plane, managing resources and routing API requests.
*   **Agent:** Manages individual server instances and launches/destroys container instances where kernels run.
*   **Storage Proxy:** Provides a unified abstraction over various network storage devices.
*   **Webserver:** Hosts the web UI for user access and administration.
*   **Kernels:** Computing environment recipes for building container images.

## Plugins

Extend Backend.AI with plugins for various functionalities, including:

*   **Accelerators:** CUDA, ROCm, and more (enterprise edition).
*   **Monitoring:** Datadog and Sentry integration.

## Getting Started

### Installation for Single-node Development

Run `scripts/install-dev.sh` after cloning this repository.

### Installation for Multi-node Tests &amp; Production

Please consult [our documentation](http://docs.backend.ai) for community-supported materials. Contact the sales team (contact@lablup.com) for professional paid support and deployment options.

## Building Packages

Build Python wheels and SCIE (Self-Contained Installable Executables) packages with the following commands:

```bash
# Build wheels or SCIE packages
./scripts/build-wheels.sh
./scripts/build-scies.sh
```

All built packages will be placed in the `dist/` directory.

## License

Refer to the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE).