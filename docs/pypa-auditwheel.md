# Auditwheel: Ensure Linux Wheel Compatibility for Python Packages

**Auditwheel simplifies the creation of compatible Python wheel packages for various Linux distributions by auditing and repairing binary extensions.** [See the original repository](https://github.com/pypa/auditwheel).

## Key Features:

*   **Auditing:** Identifies external shared library dependencies and versioned symbols in your wheel packages.
*   **Repairing:** Copies required external shared libraries into the wheel and adjusts RPATH entries for runtime compatibility, similar to static linking.
*   **Manylinux Compliance:**  Ensures your wheels adhere to the `manylinux` standards (PEP 600, 513, 571, and 599) for broader Linux distribution support.
*   **Cross-Platform Compatibility:**  Helps create wheels that run on a wider range of Linux systems.

## Overview

Auditwheel is a command-line tool designed to help developers create Python wheel packages containing pre-compiled binary extensions that are compatible with a variety of Linux distributions. It focuses on ensuring your packages meet the specifications outlined in the `PEP 600 manylinux_x_y`, `PEP 513 manylinux1`, `PEP 571 manylinux2010`, and `PEP 599 manylinux2014` platform tags.

### Commands

*   **`auditwheel show`**: Displays the external shared libraries and versioned symbols a wheel depends on. It checks the extension modules for the use of versioned symbols that exceed the `manylinux` ABI.
*   **`auditwheel repair`**: Copies necessary external shared libraries into the wheel and modifies the appropriate `RPATH` entries. This action helps to resolve dependencies at runtime.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **patchelf:** 0.14+

*Only ELF-based systems are supported.*

## Installation

Install auditwheel using pip:

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

1.  **Dynamic Library Loading:** `auditwheel` relies on `DT_NEEDED` information, so it may miss dependencies loaded via `ctypes`, `cffi`, or `dlopen`.
2.  **glibc/libstdc++ Compatibility:**  Cannot "fix" binaries compiled against overly recent `glibc` or `libstdc++` versions. Build on older Linux distributions (like manylinux Docker images) for best compatibility.

## Testing

Tests can be run with `nox`.  Integration tests may require a Docker daemon.  Run the following commands to pull and update Docker images:

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