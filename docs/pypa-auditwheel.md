# auditwheel: Ensure Compatibility for Linux Python Wheels

**auditwheel** is a powerful command-line tool designed to audit and repair Python wheel packages for broader compatibility across various Linux distributions, specifically targeting manylinux standards. [See the original repo](https://github.com/pypa/auditwheel).

## Key Features

*   **Audit Wheels:** Analyzes Python wheels to identify external shared library dependencies and versioned symbols, ensuring compliance with manylinux standards (PEP 600, PEP 513, PEP 571, PEP 599).
*   **Repair Wheels:** Copies external shared libraries into the wheel and adjusts RPATH entries, ensuring the wheel runs correctly on a wider range of Linux distributions.
*   **Manylinux Compliance:** Facilitates the creation of wheels adhering to manylinux specifications, making your Python packages more broadly accessible.
*   **Command-line Interface:** Provides straightforward commands for inspecting and repairing wheels, simplifying the process.
*   **Supports modern Python:** Requires Python 3.9 or greater.

## Overview

auditwheel is essential for creating Python wheel packages containing pre-compiled binary extensions. It ensures these packages are compatible with a wide range of Linux distributions, aligning with the manylinux standards (PEP 600, PEP 513, PEP 571, PEP 599).

### Commands

*   **`auditwheel show`**: Displays the external shared libraries a wheel depends on and checks for versioned symbols exceeding the manylinux ABI.
*   **`auditwheel repair`**: Copies necessary external shared libraries into the wheel and updates RPATH entries for runtime compatibility.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **Dependencies:** `patchelf` (version 0.14+)

## Installation

Install auditwheel using `pip`:

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

*   Dependencies loaded dynamically (using `ctypes`, `cffi`, or `dlopen`) might not be detected.
*   Cannot "fix" binaries compiled against overly recent versions of `libc` or `libstdc++`. Consider building in a manylinux Docker image for maximum compatibility.

## Testing

The tests are run with `nox`.
Some tests require Docker and images such as `quay.io/pypa/manylinux1_x86_64`.

## Code of Conduct

Please adhere to the `PSF Code of Conduct`_ when interacting with the auditwheel project.