# auditwheel: Audit and Repair Linux Wheels for Python

**Ensure your Python wheels are compatible with a wide range of Linux distributions with `auditwheel`.** ([Original Repository](https://github.com/pypa/auditwheel))

## Key Features

*   **Auditing:**  Inspects Python wheel packages to identify external shared library dependencies and versioned symbols.
*   **Repairing:**  Copies necessary shared libraries into the wheel and modifies RPATH entries for runtime compatibility.
*   **Manylinux Compliance:**  Helps create wheels compliant with PEP 600 (manylinux_x_y), PEP 513 (manylinux1), PEP 571 (manylinux2010), and PEP 599 (manylinux2014) standards.
*   **Command-Line Tool:**  Provides easy-to-use `show` and `repair` commands for wheel inspection and modification.

## Overview

`auditwheel` is a command-line tool designed to simplify the creation of Python wheel packages for Linux. These wheels often contain pre-compiled binary extensions, and `auditwheel` helps ensure they are compatible across various Linux distributions by aligning with the `manylinux` standards.  It addresses the challenge of providing pre-built binaries that work consistently across different Linux environments by identifying and resolving shared library dependencies.

## Requirements

*   **OS:** Linux
*   **Python:** 3.9+
*   **Dependency:** `patchelf` (version 0.14+)

## Installation

Install `auditwheel` using pip:

```bash
pip3 install auditwheel
```

## Examples

**Inspecting a wheel:**

```bash
auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

**Repairing a wheel:**

```bash
auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

## Limitations

*   **Dynamic Library Loading:**  `auditwheel` may miss dependencies loaded dynamically at runtime using `ctypes`, `cffi`, or `dlopen`.
*   **GLIBC/libstdc++ Compatibility:**  Cannot fix binaries compiled against newer versions of `glibc` or `libstdc++` that are incompatible with older systems. Build on older distributions or use manylinux Docker images for best compatibility.

## Testing

Run tests using `nox`. Integration tests may require a Docker daemon and specific images.

To update the Docker images, run:

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

Please adhere to the `PSF Code of Conduct`.