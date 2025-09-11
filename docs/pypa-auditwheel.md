# auditwheel: Ensure Linux Wheel Compatibility

**auditwheel is your go-to command-line tool for creating and validating Python wheel packages for Linux, ensuring compatibility across various distributions.**

[View the auditwheel repository on GitHub](https://github.com/pypa/auditwheel)

## Key Features:

*   **Auditing:** Inspects your wheel packages to identify external shared library dependencies and versioned symbols.
*   **Repairing:**  Copies necessary external libraries into the wheel and adjusts `RPATH` entries for seamless runtime compatibility, allowing you to create `manylinux` compatible wheels.
*   **Compatibility:** Supports `PEP 600 manylinux_x_y`, `PEP 513 manylinux1`, `PEP 571 manylinux2010`, and `PEP 599 manylinux2014` standards.
*   **Easy Installation:** Installs via `pip`.
*   **Integration with Docker:** Facilitates building wheels in compatible environments using manylinux Docker images.

## Overview

`auditwheel` is a command-line utility designed to help Python developers build and validate Linux wheel packages (containing pre-compiled binary extensions) that adhere to the `manylinux` standards, ensuring broad compatibility across different Linux distributions. It analyzes wheel packages to identify external shared library dependencies and checks for versioned symbols that might exceed the `manylinux` ABI.

`auditwheel` offers two primary functionalities:

*   **`auditwheel show`**: Displays external shared libraries the wheel relies on and checks for versioned symbols exceeding `manylinux` ABI limitations.
*   **`auditwheel repair`**: Copies external shared libraries into the wheel, modifies `RPATH` entries to ensure libraries are found at runtime, achieving similar results to static linking without build system modifications.

## Requirements

*   **OS:** Linux
*   **Python:** 3.9+
*   **patchelf:** 0.14+

Only systems using `ELF` (Executable and Linkable Format) based linkage are supported.

To build `manylinux` wheels, it is recommended to use pre-built manylinux Docker images.

## Installation

Install `auditwheel` using pip:

```bash
pip3 install auditwheel
```

## Examples

**Inspecting a Wheel:**

```bash
auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

**Repairing a Wheel:**

```bash
auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

## Limitations

1.  `auditwheel` relies on `DT_NEEDED` information from Python extension modules, similar to `ldd`.  Dependencies dynamically loaded via `ctypes`, `cffi`, or `dlopen` might be missed.
2.  `auditwheel` can't "fix" binaries compiled against newer versions of `libc` or `libstdc++` due to symbol versioning.  For maximum compatibility, build on an older Linux distribution, like a manylinux Docker image.

## Testing

Tests can be executed using `nox`. Some integration tests require a running Docker daemon. You may need to pull certain Docker images manually to update them.

## Code of Conduct

All contributors are expected to adhere to the `PSF Code of Conduct`.