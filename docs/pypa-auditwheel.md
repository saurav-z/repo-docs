# auditwheel: Ensure Cross-Platform Compatibility for Python Wheels

**auditwheel** is your go-to command-line tool for creating Python wheels that seamlessly work across various Linux distributions, adhering to the manylinux standards.  [Learn more at the original repository](https://github.com/pypa/auditwheel).

## Key Features

*   **Inspects Wheel Dependencies:** Analyzes Python wheels to identify external shared libraries and versioned symbols that could cause compatibility issues.
*   **Repairs Wheels:**  Copies external shared libraries into the wheel and modifies RPATH entries, making your wheels compatible with a wider range of Linux distributions without modifying the build system.
*   **Supports Manylinux Standards:**  Works with `PEP 600 manylinux_x_y`, `PEP 513 manylinux1`, `PEP 571 manylinux2010`, and `PEP 599 manylinux2014` platform tags.
*   **Simple Command-Line Interface:** Provides easy-to-use commands for inspection (`auditwheel show`) and repair (`auditwheel repair`).

## Overview

Auditwheel streamlines the process of building and distributing Python wheel packages containing pre-compiled binary extensions for Linux. It addresses the challenge of ensuring compatibility across different Linux distributions by:

*   Identifying dependencies on external shared libraries that might not be available on all target systems.
*   "Repairing" wheels by bundling necessary shared libraries and adjusting runtime paths (RPATH) to ensure they are found.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **Dependencies:** `patchelf` (version 0.14+)

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

*   **Dynamic Library Loading:**  Dependencies loaded dynamically via `ctypes`, `cffi`, or `dlopen` might be missed.
*   **`libc` and `libstdc++` Versioning:** Auditwheel cannot fix binaries compiled against overly recent versions of `libc` or `libstdc++`.  Building in a manylinux Docker image is recommended for maximum compatibility.

## Testing

Run tests using `nox`. Some integration tests require a running Docker daemon.

**Docker Image Updates:**

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

Please adhere to the `PSF Code of Conduct`_ when interacting with the `auditwheel` project.