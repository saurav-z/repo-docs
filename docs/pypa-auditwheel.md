# auditwheel: Ensure Compatibility for Python Wheels on Linux

**auditwheel is a powerful command-line tool that helps you create and verify Python wheels compatible with various Linux distributions, adhering to manylinux standards.** ([See the original repo](https://github.com/pypa/auditwheel))

## Key Features:

*   **Auditing Wheels:** Inspects Python wheel packages to identify external shared library dependencies and compatibility issues based on PEP 600, PEP 513, PEP 571, and PEP 599 manylinux standards.
*   **Dependency Analysis:**  Shows external shared libraries that the wheel depends on, and checks for the use of versioned symbols.
*   **Repairing Wheels:**  Copies external shared libraries into the wheel and modifies RPATH entries to ensure runtime compatibility.
*   **Platform Tag Management:**  Correctly modifies the wheel's platform tags to reflect manylinux compliance.

## Overview

`auditwheel` simplifies the creation of Python wheel packages for Linux, especially those containing pre-compiled binary extensions.  It focuses on ensuring these wheels are compatible across a wide range of Linux distributions, aligning with the manylinux policies (PEP 600, PEP 513, PEP 571, PEP 599).  This tool is essential for packaging Python extensions that need to run seamlessly on different Linux systems.

### Functionality

*   **`auditwheel show`**: Displays external shared libraries and versioned symbols a wheel depends on. It verifies compliance with manylinux standards.
*   **`auditwheel repair`**: Copies external shared libraries into the wheel and updates RPATH entries for correct runtime behavior.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **Dependencies:** `patchelf <https://github.com/NixOS/patchelf>`_ (version 0.14+)
*   **ELF Support:**  Requires an ELF-based system (virtually all Linux distributions).

## Installation

Install `auditwheel` easily using pip:

```bash
pip3 install auditwheel
```

## Examples

**Inspect a wheel:**

```bash
auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

This will show the wheel's dependencies and any compatibility issues.

**Repair a wheel:**

```bash
auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

This will fix the wheel to be manylinux compliant, copying necessary libraries and updating paths.

## Limitations

*   **Dynamic Library Loading:**  Dependencies loaded using `ctypes`, `cffi` (Python), or `dlopen` (C/C++) may not be detected.
*   **Compiler Compatibility:**  `auditwheel` cannot fix issues from binaries compiled with excessively new versions of `glibc` or `libstdc++`. Building on older Linux distributions (e.g., within manylinux Docker images) is recommended.

## Testing

Run tests using `nox`. Some integration tests require a Docker daemon.

To update Docker images:

```bash
docker pull python:3.9-slim-bookworm
docker pull quay.io/pypa/manylinux1_x86_64
docker pull quay.io/pypa/manylinux2010_x86_64
docker pull quay.io/pypa/manylinux2014_x86_64
docker pull quay.io/pypa/manylinux_2_28_x86_64
docker pull quay.io/pypa/manylinux_2_34_x86_64
docker pull quay.io/pypa/musllinux_1_2_x86_64
```

You can remove Docker images with `docker rmi`.

## Code of Conduct

This project adheres to the `PSF Code of Conduct <https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md>`.