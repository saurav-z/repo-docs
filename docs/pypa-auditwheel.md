# Auditwheel: Ensure Linux Wheel Compatibility for Python Packages

**Auditwheel** is a powerful command-line tool that helps you create and audit Python wheel packages for Linux, ensuring they are compatible with a wide range of distributions. ([Original Repository](https://github.com/pypa/auditwheel))

## Key Features:

*   **Auditing:** Analyzes Python wheels to identify external shared library dependencies and potential compatibility issues related to manylinux standards (PEP 600, PEP 513, PEP 571, and PEP 599).
*   **Repairing:** Modifies wheels by bundling required external shared libraries and adjusting runtime paths (RPATH) to ensure they can be found, simplifying distribution.
*   **Platform Tagging:** Correctly updates the wheel's platform tags to reflect manylinux compatibility (e.g., `manylinux1_x86_64`).
*   **Easy Integration:** Simple command-line interface for easy inspection and modification of wheels.

## Overview

Auditwheel enables the creation of Python wheel packages containing pre-compiled binary extensions that adhere to the manylinux standards. These standards define a set of compatible Linux distributions, making your packages more widely accessible.

### `auditwheel show`:

*   Displays external shared libraries a wheel depends on.
*   Checks extension modules for versioned symbols exceeding manylinux ABI.

### `auditwheel repair`:

*   Copies external shared libraries into the wheel.
*   Modifies RPATH entries so these libraries are found at runtime.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **Dependencies:** `patchelf <https://github.com/NixOS/patchelf>`_ (version 0.14+)

## Installation

Install auditwheel using pip:

```bash
pip3 install auditwheel
```

## Examples

### Inspecting a Wheel

```bash
auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

This will show the dependencies and compatibility details of the wheel.

### Repairing a Wheel

```bash
auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

This command will fix the wheel to be compatible with manylinux standards.

## Limitations

*   **Dynamic Library Loading:**  Dependencies loaded via `ctypes`, `cffi`, or `dlopen` might be missed.
*   **GLIBC/libstdc++ Compatibility:**  Cannot fix issues if binaries are compiled against a too-recent version of `glibc` or `libstdc++`. Build on an older Linux distribution (e.g., manylinux Docker image) for best compatibility.

## Testing

Run tests with `nox`. Integration tests require a Docker daemon. Use `docker pull` to update images if needed.

## Code of Conduct

Please adhere to the `PSF Code of Conduct <https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md>`_ when interacting with the auditwheel project.