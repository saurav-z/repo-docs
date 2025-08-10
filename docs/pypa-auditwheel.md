# auditwheel: Ensuring Linux Wheel Compatibility for Python Packages

**Auditwheel is a powerful command-line tool designed to audit and repair Python wheel packages for compatibility with various Linux distributions, specifically those adhering to PEP 600 manylinux standards.** [(See original repo)](https://github.com/pypa/auditwheel)

## Key Features

*   **Auditing:** Identifies external shared library dependencies and versioned symbols in your wheel packages.
*   **Repairing:** Copies required external shared libraries into the wheel and modifies RPATH entries, ensuring compatibility across a wider range of Linux systems.
*   **Manylinux Compliance:** Supports PEP 600 (manylinux_x_y), PEP 513 (manylinux1), PEP 571 (manylinux2010), and PEP 599 (manylinux2014) wheel standards.
*   **Platform Tag Management:** Updates wheel platform tags to reflect manylinux compatibility (e.g., from `linux_x86_64` to `manylinux1_x86_64`).
*   **Command-Line Interface:** Provides easy-to-use commands like `show` (for inspection) and `repair` (for fixing wheels).

## Overview

Auditwheel simplifies the creation of cross-platform Python wheel packages that include pre-compiled binary extensions. It helps developers ensure their packages work on diverse Linux distributions.

### Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **patchelf:** 0.14+ (Get it from [here](https://github.com/NixOS/patchelf))

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

*   **Dynamic Library Loading:** Does not detect dependencies loaded dynamically at runtime (via `ctypes`, `cffi`, or `dlopen`).
*   **glibc/libstdc++ Compatibility:** Cannot "fix" binaries compiled against overly recent versions of `glibc` or `libstdc++` due to symbol versioning issues. Building on an older Linux distribution is recommended.

## Testing

Test `auditwheel` by using ``nox``.

To update the test Docker images:

```bash
docker pull python:3.9-slim-bookworm
docker pull quay.io/pypa/manylinux1_x86_64
docker pull quay.io/pypa/manylinux2010_x86_64
docker pull quay.io/pypa/manylinux2014_x86_64
docker pull quay.io/pypa/manylinux_2_28_x86_64
docker pull quay.io/pypa/manylinux_2_34_x86_64
docker pull quay.io/pypa/musllinux_1_2_x86_64
```

You may remove these images using ``docker rmi``.

## Code of Conduct

Please adhere to the [PSF Code of Conduct](https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md).