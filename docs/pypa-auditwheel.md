# Auditwheel: Ensure Compatibility of Python Wheels on Linux

**Auditwheel is a crucial command-line tool for creating Python wheels that are compatible across various Linux distributions, especially those adhering to PEP 600 (manylinux).**  Check out the [original repo here](https://github.com/pypa/auditwheel).

## Key Features

*   **Auditing:** Analyzes Python wheel packages to identify external shared library dependencies.
*   **Compatibility Checks:** Verifies that extension modules meet the requirements of the manylinux policies (PEP 600, 513, 571, and 599).
*   **Repairing:** Copies necessary external shared libraries into the wheel, modifying RPATH entries to ensure proper runtime linking, similar to static linking but without modifying the build system.
*   **Platform Tagging:**  Correctly updates wheel tags (e.g., manylinux1, manylinux2010, manylinux2014) to reflect compatibility.

## Overview

Auditwheel streamlines the creation of Python wheel packages for Linux, particularly those containing compiled binary extensions. These wheels, once processed by `auditwheel`, are designed to work seamlessly across a wide range of Linux distributions. It helps in making your Python packages more accessible and usable by a larger audience.

### Key Functionality

*   **`auditwheel show`:**
    *   Displays external shared library dependencies.
    *   Checks extension modules for versioned symbols exceeding the `manylinux` ABI.
*   **`auditwheel repair`:**
    *   Bundles external shared libraries within the wheel.
    *   Adjusts `RPATH` entries for correct library loading at runtime.
    *   Updates wheel platform tags to reflect manylinux compliance.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **patchelf:** 0.14+ (required for modifying binaries)

## Installation

Install `auditwheel` using pip:

```bash
pip3 install auditwheel
```

## Examples

### Inspecting a Wheel

```bash
auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

This command will output information about the wheel's external dependencies, versioned symbols, and potential compatibility issues.

### Repairing a Wheel

```bash
auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

This command fixes the wheel to make it manylinux-compatible.

## Limitations

*   **Dynamic Library Loading:**  Dependencies loaded dynamically at runtime via `ctypes`, `cffi`, or `dlopen` may not be detectable.
*   **Binary Compilation:** Auditwheel cannot fix binaries compiled against overly recent versions of `libc` or `libstdc++`. Build on older Linux distributions for wider compatibility.

## Testing

The tests can be run with ``nox``.

## Code of Conduct

The `auditwheel` project adheres to the [PSF Code of Conduct](https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md).