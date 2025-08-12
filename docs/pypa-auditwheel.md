# auditwheel: Audit and Repair Linux Wheels for Python

**auditwheel** is a powerful command-line tool designed to audit, inspect, and repair Python wheel packages, ensuring broad compatibility across various Linux distributions, especially for packages with compiled binary extensions.  [Learn more on GitHub](https://github.com/pypa/auditwheel).

## Key Features

*   **Auditing:** Identifies external shared library dependencies within your wheel packages that may limit compatibility.
*   **Repairing:** Copies necessary shared libraries into the wheel and adjusts `RPATH` entries for seamless runtime operation, enabling broader distribution compatibility.
*   **manylinux Compliance:**  Supports the creation of wheels compliant with `PEP 600`, `PEP 513`, `PEP 571`, and `PEP 599` manylinux standards for maximum cross-platform compatibility.
*   **Inspection:** Provides detailed insights into wheel dependencies and potential compatibility issues using the `auditwheel show` command.
*   **Wide Linux Support:** Compatible with any Linux system that uses ELF-based linkage.

## Overview

`auditwheel` streamlines the process of building compatible Python wheel packages for Linux. It's crucial for developers who need to distribute binary extensions to ensure their packages work flawlessly across various Linux distributions.

## Requirements

*   **OS:** Linux
*   **Python:** 3.9+
*   **patchelf:** 0.14+ ([https://github.com/NixOS/patchelf](https://github.com/NixOS/patchelf))

## Installation

Install `auditwheel` using pip:

```bash
pip3 install auditwheel
```

## Examples

**Inspect a wheel:**

```bash
auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

**Repair a wheel:**

```bash
auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

## Limitations

*   Dependencies loaded dynamically at runtime via `ctypes`, `cffi`, or `dlopen` might not be detected.
*   `auditwheel` cannot fix binaries compiled against excessively recent versions of `libc` or `libstdc++`.

## Testing

Tests can be run with `nox`. Some integration tests require Docker. You can update Docker images with:

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

Please adhere to the `PSF Code of Conduct`_ when interacting with this project.