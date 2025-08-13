# Auditwheel: Ensuring Linux Wheel Compatibility for Python Packages

Auditwheel is a command-line tool designed to simplify the creation of compatible Python wheels for Linux, ensuring they meet the standards of `PEP 600 manylinux_x_y <https://www.python.org/dev/peps/pep-0600/>`, `PEP 513 manylinux1 <https://www.python.org/dev/peps/pep-0513/>`, `PEP 571 manylinux2010 <https://www.python.org/dev/peps/pep-0571/>` and `PEP 599 manylinux2014 <https://www.python.org/dev/peps/pep-0599/>`.

**[View the original repository](https://github.com/pypa/auditwheel)**

## Key Features

*   **Auditing:** Identifies external shared library dependencies and potential compatibility issues within your wheel packages.
*   **Repairing:** Bundles necessary shared libraries within the wheel, adjusting `RPATH` entries to ensure they are correctly loaded at runtime, making your wheels more widely compatible.
*   **Compatibility Focus:** Supports the creation of wheels compliant with various `manylinux` standards, expanding the reach of your Python packages.
*   **Command-Line Interface:** Offers a straightforward command-line interface for easy inspection and repair of wheels.

## Overview

Auditwheel helps developers create Python wheels (containing pre-compiled binary extensions) that run on various Linux distributions.  It addresses the complexities of binary compatibility by analyzing and modifying wheel packages to adhere to `manylinux` standards. The tool analyzes dependencies and, through the "repair" functionality, ensures that the wheel includes all the necessary shared libraries, correctly configured for runtime loading.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **Dependency:** `patchelf <https://github.com/NixOS/patchelf>`_ (version 0.14+)

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

*   Auditwheel relies on the information provided by `DT_NEEDED`, which may not cover all runtime dependencies.
*   It can't fix binaries compiled against too-recent versions of `libc` or `libstdc++`.

## Testing

The tests can be run with `nox`. Some tests require a running Docker daemon.

## Code of Conduct

Please adhere to the `PSF Code of Conduct <https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md>`_ when interacting within this project.