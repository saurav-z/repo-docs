# auditwheel: Ensure Linux Wheel Compatibility

**auditwheel is a powerful command-line tool designed to audit and repair Python wheel packages, ensuring compatibility across various Linux distributions.**  ([See the original repository](https://github.com/pypa/auditwheel))

## Key Features

*   **Auditing:** Analyzes your wheel packages to identify external shared library dependencies and potential compatibility issues with manylinux standards (PEP 513, PEP 571, PEP 599, and PEP 600).
*   **Repairing:**  Copies required external libraries into the wheel and modifies RPATH entries, making your wheels more self-contained and compatible with a wider range of Linux systems.
*   **Compliance:** Facilitates the creation of Python wheels adhering to manylinux standards.
*   **Simplifies Distribution:** Reduces the need for complex build configurations and ensures that your binary extensions work seamlessly across different Linux distributions.

## Overview

auditwheel helps you create compliant Python wheels for Linux containing pre-compiled binary extensions, adhering to the manylinux standards. It achieves this by:

*   **Inspecting Wheels:**  The `auditwheel show` command reveals external shared library dependencies, identifying potential compatibility problems.
*   **Fixing Dependencies:** The `auditwheel repair` command copies external shared libraries into the wheel and adjusts RPATH entries.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **patchelf:** 0.14+ (required for modifying the wheels)

Only systems that use `ELF <https://en.wikipedia.org/wiki/Executable_and_Linkable_Format>`_-based linkage are supported.

## Installation

Install auditwheel easily using pip:

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

*   Dependencies loaded dynamically via `ctypes`, `cffi` or `dlopen` might not be detected.
*   It cannot fix issues if compiled with a too-recent version of `libc` or `libstdc++`. It's best to build on an older Linux distribution (e.g., manylinux Docker image).

## Testing

Tests can be run with `nox`. Some integration tests need Docker.  Update these images with the docker pull commands in the original README.

## Code of Conduct

This project follows the `PSF Code of Conduct`.