# Auditwheel: Ensure Linux Wheel Compatibility for Python Packages

**Auditwheel is a powerful command-line tool designed to audit and repair Python wheel packages for broader Linux distribution compatibility, adhering to manylinux standards.** [View the original repository](https://github.com/pypa/auditwheel).

## Key Features

*   **Auditing:**  Analyzes Python wheel packages to identify external shared library dependencies and compatibility with manylinux standards (PEP 600, PEP 513, PEP 571, PEP 599).
*   **Repairing:** Modifies wheels to include necessary external shared libraries, automatically adjusting RPATH entries for seamless runtime operation, ensuring broader Linux distribution compatibility.
*   **manylinux Compliance:** Supports manylinux1, manylinux2010, manylinux2014, and other manylinux standards for compatibility.
*   **Dependency Inspection:** Shows external shared libraries a wheel depends on, and checks the extension modules for the use of versioned symbols that exceed the manylinux ABI.
*   **Easy Installation:** Install with pip: `pip3 install auditwheel`.

## Overview

Auditwheel streamlines the process of creating and distributing Python wheels that contain pre-compiled binary extensions, ensuring compatibility across a wide range of Linux distributions. It is essential for Python developers aiming to provide pre-built binary packages for Linux users.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **patchelf:** 0.14+ ([https://github.com/NixOS/patchelf](https://github.com/NixOS/patchelf))

## Installation

Install auditwheel using pip:

```bash
pip3 install auditwheel
```

## Examples

### Inspecting a Wheel:

```bash
auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

This command will display the wheel's platform tag, external dependencies, and any issues preventing manylinux compliance.

### Repairing a Wheel:

```bash
auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

This command repairs the wheel, incorporating the necessary external libraries and updating the platform tag for manylinux compatibility.

## Limitations

*   **Dynamic Library Loading:** Auditwheel may miss dependencies loaded dynamically via `ctypes`, `cffi` (Python) or `dlopen` (C/C++).
*   **glibc/libstdc++ Versioning:** Compatibility issues can arise when binaries are compiled against newer versions of `glibc` or `libstdc++`. Building within a manylinux Docker image is recommended to ensure maximum compatibility.

## Testing

Run tests using `nox`.  Some integration tests also require a running Docker daemon.

To update Docker images used for testing:

```bash
docker pull python:3.9-slim-bookworm
docker pull quay.io/pypa/manylinux1_x86_64
docker pull quay.io/pypa/manylinux2010_x86_64
docker pull quay.io/pypa/manylinux2014_x86_64
docker pull quay.io/pypa/manylinux_2_28_x86_64
docker pull quay.io/pypa/manylinux_2_34_x86_64
docker pull quay.io/pypa/musllinux_1_2_x86_64
```

## Code of Conduct

The project follows the `PSF Code of Conduct`_.