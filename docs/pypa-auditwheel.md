# auditwheel: Make Your Python Wheels Linux-Compatible

**Ensure your Python wheels are compatible with a wide range of Linux distributions with `auditwheel`.**  [View the original repo](https://github.com/pypa/auditwheel)

## Key Features

*   **Auditing:**  Analyze Python wheel packages to identify external shared library dependencies and compatibility issues with manylinux standards.
*   **Repairing:**  Automatically bundle required shared libraries within your wheel and adjust runtime paths (RPATH) to ensure they are found, simplifying deployment across diverse Linux environments.
*   **Compliance:**  Supports PEP 600 (manylinux_x_y), PEP 513 (manylinux1), PEP 571 (manylinux2010), and PEP 599 (manylinux2014) for broader compatibility.
*   **Easy to Use:** Command-line tool simplifies the process of inspecting and fixing your Python wheels.
*   **Platform Support:**  Designed for Linux systems using ELF-based linkage.

## Overview

`auditwheel` is a command-line tool designed to help create Python wheel packages containing pre-compiled binary extensions that are compatible with various Linux distributions. It focuses on compliance with manylinux standards, ensuring your packages run smoothly on a broad range of systems.

### `auditwheel show`

*   Displays external shared libraries a wheel depends on.
*   Checks extension modules for versioned symbols exceeding manylinux ABI.

### `auditwheel repair`

*   Copies external shared libraries into the wheel.
*   Modifies RPATH entries for correct runtime library loading.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **Dependency:** `patchelf <https://github.com/NixOS/patchelf>`_ (version 0.14+)

## Installation

Install `auditwheel` using pip:

```bash
pip3 install auditwheel
```

## Examples

### Inspecting a Wheel:

```bash
auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

### Repairing a Wheel:

```bash
auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

## Limitations

1.  Doesn't detect dependencies loaded dynamically through `ctypes`, `cffi` or `dlopen`.
2.  Cannot fix binaries compiled against overly recent versions of `libc` or `libstdc++`.

## Testing

Use `nox` to run tests, which will handle test dependency installations. Some integration tests require a Docker daemon. Update Docker images manually:

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

The project follows the [PSF Code of Conduct](https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md).