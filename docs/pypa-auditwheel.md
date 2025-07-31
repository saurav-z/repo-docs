# auditwheel: Ensure Linux Wheel Compatibility for Python Packages

**auditwheel** is a powerful command-line tool designed to help you create and audit Python wheel packages for Linux, ensuring they are compatible with a wide range of Linux distributions. Check out the original repository for more details: [https://github.com/pypa/auditwheel](https://github.com/pypa/auditwheel).

## Key Features

*   **Auditing:** Analyzes Python wheels to identify external shared library dependencies and compatibility with manylinux standards (PEP 600, 513, 571, and 599).
*   **Repairing:** Bundles necessary shared libraries within the wheel and modifies the `RPATH` entries, making your wheels more portable without build system modifications.
*   **Manylinux Compatibility:** Simplifies the creation of wheels conforming to manylinux standards, crucial for widespread compatibility across different Linux distributions.
*   **Show Functionality:** Provides a detailed view of a wheel's external dependencies and potential compatibility issues.

## Overview

auditwheel streamlines the process of building compatible Python wheel packages containing pre-compiled binary extensions for Linux.  It supports the `PEP 600 manylinux_x_y <https://www.python.org/dev/peps/pep-0600/>`, `PEP 513 manylinux1 <https://www.python.org/dev/peps/pep-0513/>`, `PEP 571 manylinux2010 <https://www.python.org/dev/peps/pep-0571/>` and `PEP 599 manylinux2014 <https://www.python.org/dev/peps/pep-0599/>`  platform tags.

## Requirements

*   OS: Linux
*   Python: 3.9+
*   `patchelf <https://github.com/NixOS/patchelf>`_: 0.14+

## Installation

Install `auditwheel` using pip:

```bash
pip3 install auditwheel
```

## Example Usage

**Inspecting a wheel:**

```bash
$ auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

This command will display information about the wheel's dependencies, compatibility, and any necessary adjustments.

**Repairing a wheel:**

```bash
$ auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

This will repair the wheel by bundling missing dependencies and adjusting the RPATH to ensure compatibility with manylinux standards.

## Limitations

*   **Dynamic Loading:**  Dependencies loaded dynamically at runtime via `ctypes`, `cffi`, or `dlopen` may not be detected.
*   **Compiler Compatibility:**  Cannot fix binaries linked against overly new versions of `glibc` or `libstdc++`, requiring builds on older Linux distributions (e.g., manylinux Docker images) for maximum compatibility.

## Testing

Run tests with `nox`.

Some integration tests require Docker.  To manually update Docker images:

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

Please adhere to the `PSF Code of Conduct <https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md>`_ when interacting with this project.