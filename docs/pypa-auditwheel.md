# Auditwheel: Ensure Linux Wheel Compatibility for Python Packages

Auditwheel is a powerful command-line tool designed to audit, repair, and relabel Python wheel packages for Linux, ensuring compatibility across a wide range of distributions. You can find the original repository [here](https://github.com/pypa/auditwheel).

## Key Features

*   **Auditing:** Analyze Python wheels to identify external shared library dependencies and potential compatibility issues with `manylinux` standards (PEP 600, PEP 513, PEP 571, and PEP 599).
*   **Repairing:** Copies necessary external shared libraries into the wheel and automatically adjusts the `RPATH` entries to resolve runtime dependencies.
*   **Relabeling:** Modifies wheel filenames and metadata to comply with `manylinux` platform tags for broader compatibility.
*   **Extensive Platform Support:** Supports manylinux1, manylinux2010, manylinux2014 and newer standards.

## Overview

Auditwheel is a crucial tool for Python developers who need to distribute pre-compiled binary extensions for Linux. It streamlines the process of creating `wheel packages` that are compatible with diverse Linux distributions, adhering to the `PEP 600 manylinux_x_y`, `PEP 513 manylinux1`, `PEP 571 manylinux2010`, and `PEP 599 manylinux2014` platform tags.

### Usage

*   **`auditwheel show`:** Displays external shared libraries required by a wheel and checks for versioned symbols exceeding the `manylinux` ABI.
*   **`auditwheel repair`:** Copies external shared libraries into the wheel and modifies `RPATH` entries to ensure they are found at runtime.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **Dependencies:**
    *   `patchelf <https://github.com/NixOS/patchelf>`_: 0.14+
    *   Only systems that use `ELF <https://en.wikipedia.org/wiki/Executable_and_Linkable_Format>`_-based linkage are supported

## Installation

Install `auditwheel` using pip:

```bash
pip3 install auditwheel
```

## Examples

**Inspecting a Wheel:**

```bash
$ auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

**Repairing a Wheel:**

```bash
$ auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

## Limitations

1.  Dependencies loaded dynamically through `ctypes`, `cffi`, or `dlopen` are not always detected.
2.  Binaries compiled against newer versions of `libc` or `libstdc++` might not be compatible with older systems.

## Testing

Run tests with `nox`.  Integration tests may require a running Docker daemon and the following images:

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

This project follows the `PSF Code of Conduct`_.