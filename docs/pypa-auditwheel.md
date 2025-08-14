# auditwheel: Ensure Linux Wheel Compatibility for Python Packages

**Auditwheel is a powerful command-line tool for creating and modifying Python wheel packages to ensure compatibility across various Linux distributions, adhering to manylinux standards.**  ([Original Repo](https://github.com/pypa/auditwheel))

## Key Features

*   **Wheel Inspection:**  Analyze wheels to identify external shared library dependencies and compatibility issues related to manylinux standards (PEP 600, 513, 571, and 599).
*   **Wheel Repair:**  Automatically modifies wheels by bundling necessary shared libraries and updating `RPATH` entries, ensuring runtime compatibility across different Linux systems.
*   **Manylinux Compliance:**  Helps create wheels that conform to manylinux platform tags, increasing the distribution and usability of your Python packages.
*   **Versioning Support:**  Checks extension modules for versioned symbols to determine compatibility.
*   **Easy Installation:** Install with a simple `pip install auditwheel` command.

## Overview

Auditwheel streamlines the process of building Python wheel packages containing pre-compiled binary extensions for Linux, ensuring broad compatibility.  It focuses on compliance with manylinux standards, which specify the minimum system requirements for wheels, allowing them to run on a wide array of Linux distributions.

### Functionality:

*   `auditwheel show`:  Displays external shared library dependencies, highlighting potential compatibility issues, and checks for the use of versioned symbols.
*   `auditwheel repair`:  Copies external shared libraries into the wheel and adjusts `RPATH` entries to ensure they're found at runtime, effectively resolving dependency conflicts and creating a more compatible wheel.

## Requirements

*   **OS:** Linux
*   **Python:** 3.9+
*   **patchelf:** 0.14+ (for modifying the binaries - [GitHub Repo](https://github.com/NixOS/patchelf))

**Note:**  Auditwheel supports only ELF-based linkage, common to all Linux distributions.

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

1.  **Dynamic Library Loading:** Auditwheel's analysis relies on `DT_NEEDED` information, missing dependencies loaded dynamically via `ctypes`, `cffi`, or `dlopen`.
2.  **Compiler Compatibility:** It cannot fix binaries built against newer versions of `libc` or `libstdc++`. Building on an older Linux distribution (such as a manylinux Docker image) is recommended for wider compatibility.

## Testing

Run tests using `nox`.  Some tests require a running Docker daemon.  To update Docker images manually, run the commands in the original README.

## Code of Conduct

Please adhere to the `PSF Code of Conduct`_ when interacting with this project.