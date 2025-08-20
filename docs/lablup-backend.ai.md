# Backend.AI: A Powerful, Container-Based Computing Platform

[Backend.AI](https://github.com/lablup/backend.ai) is a cutting-edge platform that simplifies and streamlines the execution of computationally intensive tasks. 

[![PyPI release version](https://badge.fury.io/py/backend.ai-manager.svg)](https://pypi.org/project/backend.ai-manager/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/backend.ai-manager.svg)
![Wheels](https://img.shields.io/pypi/wheel/backend.ai-manager.svg)
[![Gitter](https://badges.gitter.im/lablup/backend.ai.svg)](https://gitter.im/lablup/backend.ai)

**Key Features:**

*   **Container-Based Computing:** Effortlessly run your code in isolated containers.
*   **Multi-Framework Support:** Supports popular computing and machine learning frameworks.
*   **Heterogeneous Accelerator Support:**  Includes support for CUDA GPUs, ROCm GPUs, and more.
*   **REST and GraphQL APIs:**  Provides comprehensive API access for easy integration.
*   **On-Demand & Batch Computation:** Supports both interactive and batch job execution.
*   **Secure Access:** Provides secure websocket tunneling into computation sessions.
*   **Web-Based Access:** Jupyter, Web-based terminal, SSH, and VSCode support.
*   **Virtual Folder (vfolder) Support:** An abstraction layer over existing network-based storages (e.g., NFS/SMB) that is mounted into any computation sessions.
*   **Client SDKs**:  Easy-to-use SDKs are available in Python, Java, Javascript, and PHP.

## Core Components

### Manager
The control plane orchestrates the cluster and manages API requests.
*   [Manager README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/manager/README.md)
*   Handles API requests and manages/scales the cluster.

### Agent
The agent manages individual server instances and launches/destroys Docker containers where REPL daemons (kernels) run.
*   [Agent README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/agent/README.md)
*   Manages individual server instances.

### Storage Proxy
Provides a unified abstraction over various network storage devices.
*   [Storage Proxy README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/storage/README.md)

### Webserver
Hosts the Web UI for end-users and administration tasks.
*   [Webserver README](https://github.com/lablup/backend.ai/blob/main/src/ai/backend/web/README.md)

### Kernels
Computing environment recipes (Dockerfile) to build the container images.
*   [Kernels](https://github.com/lablup/backend.ai-kernels)

### Jail
A programmable sandbox.
*   [Jail](https://github.com/lablup/backend.ai-jail)

### Hook
A set of libc overrides for resource control and web-based interactive stdin.
*   [Hook](https://github.com/lablup/backend.ai-hook)

## Client SDK Libraries

SDKs are available to ease integration with software products:
*   Python: `pip install backend.ai-client` ([Python SDK](https://github.com/lablup/backend.ai/tree/main/src/ai/backend/client))
*   Java: ([Java SDK](https://github.com/lablup/backend.ai-client-java))
*   Javascript: `npm install backend.ai-client` ([Javascript SDK](https://github.com/lablup/backend.ai-client-js))
*   PHP: `composer require lablup/backend.ai-client` ([PHP SDK](https://github.com/lablup/backend.ai-client-php))

## Plugins

Backend.AI's plugin architecture allows for easy extension of functionality.

### Accelerator Plugins
*   `backendai_accelerator_v21`
    *   `ai.backend.accelerator.cuda`: CUDA accelerator plugin
    *   `ai.backend.accelerator.cuda` (mock): CUDA mockup plugin
    *   `ai.backend.accelerator.rocm`

### Monitoring Plugins
*   `backendai_monitor_stats_v10`
    *   `ai.backend.monitor.stats` based on the Datadog API
*   `backendai_monitor_error_v10`
    *   `ai.backend.monitor.error` based on the Sentry API

## Getting Started

### Installation for Single-node Development

Run `scripts/install-dev.sh` after cloning this repository.

This script checks availability of all required dependencies such as Docker and bootstrap a development setup.  Note that it requires `sudo` and a modern Python installed in the host system based on Linux (Debian/RHEL-likes) or macOS.

### Installation for Multi-node Tests &amp; Production

Please consult [our documentation](http://docs.backend.ai) for community-supported materials.
Contact the sales team (contact@lablup.com) for professional paid support and deployment options.

### Accessing Compute Sessions (aka Kernels)

Backend.AI provides websocket tunneling into individual computation sessions (containers),
so that users can use their browsers and client CLI to access in-container applications directly
in a secure way.

* Jupyter: data scientists' favorite tool
   * Most container images have intrinsic Jupyter and JupyterLab support.
* Web-based terminal
   * All container sessions have intrinsic ttyd support.
* SSH
   * All container sessions have intrinsic SSH/SFTP/SCP support with auto-generated per-user SSH keypair.
     PyCharm and other IDEs can use on-demand sessions using SSH remote interpreters.
* VSCode
   * Most container sessions have intrinsic web-based VSCode support.

### Working with Storage

Backend.AI provides an abstraction layer on top of existing network-based storages
(e.g., NFS/SMB), called vfolders (virtual folders).
Each vfolder works like a cloud storage that can be mounted into any computation
sessions and shared between users and user groups with differentiated privileges.


## Building Packages

To build packages, run:

```bash
./scripts/build-wheels.sh  # Build wheels
./scripts/build-scies.sh   # Build SCIE packages
```
All built packages will be placed in the `dist/` directory.

## Python Version Compatibility

| Backend.AI Core Version | Python Version | Pantsbuild version |
|:-----------------------:|:--------------:|:------------------:|
| 25.06.x ~               | 3.13.x         | 2.23.x             |
| 24.03.x / 24.09.x ~ 25.05.x      | 3.12.x         | 2.21.x    |
| 23.03.x / 23.09.x       | 3.11.x         | 2.19.x             |
| 22.03.x / 22.09.x       | 3.10.x         |                    |
| 21.03.x / 21.09.x       | 3.8.x          |                    |

## License

Backend.AI is licensed under LGPLv3 for server-side components and MIT license for shared libraries/SDKs.  See the [LICENSE file](https://github.com/lablup/backend.ai/blob/main/LICENSE) for details.