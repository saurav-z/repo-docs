# Auditwheel: Ensure Compatibility for Python Wheels on Linux

Auditwheel is a powerful command-line tool designed to create and audit Python wheel packages for Linux, ensuring compatibility across various distributions by adhering to [PEP 600](https://www.python.org/dev/peps/pep-0600/), [PEP 513](https://www.python.org/dev/peps/pep-0513/), [PEP 571](https://www.python.org/dev/peps/pep-0571/), and [PEP 599](https://www.python.org/dev/peps/pep-0599/) standards.

For more details, see the original repository: [https://github.com/pypa/auditwheel](https://github.com/pypa/auditwheel)

## Key Features

*   **Inspection:** Shows external shared library dependencies and checks for versioned symbols, ensuring compliance with manylinux policies.
*   **Repair:** Copies necessary external shared libraries into the wheel and modifies `RPATH` entries to ensure they are found at runtime. This facilitates wider compatibility without requiring build system changes.
*   **Manylinux Compliance:** Helps create wheels compatible with manylinux1, manylinux2010, manylinux2014, and other manylinux standards.

## Overview

Auditwheel facilitates the creation of Python wheel packages for Linux (containing pre-compiled binary extensions) that are compatible with a wide variety of Linux distributions, consistent with the `PEP 600 manylinux_x_y <https://www.python.org/dev/peps/pep-0600/>`_, `PEP 513 manylinux1 <https://www.python.org/dev/peps/pep-0513/>`_, `PEP 571 manylinux2010 <https://www.python.org/dev/peps/pep-0571/>`_ and `PEP 599 manylinux2014 <https://www.python.org/dev/peps/pep-0599/>`_ platform tags.

## Installation

Install auditwheel using pip:

```bash
pip3 install auditwheel
```

## Examples

*   **Inspecting a wheel:**

```bash
$ auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

*   **Repairing a wheel:**

```bash
$ auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

## Requirements

*   OS: Linux
*   Python: 3.9+
*   `patchelf <https://github.com/NixOS/patchelf>`_: 0.14+

## Limitations

1.  Dependencies loaded dynamically via `ctypes`, `cffi`, or `dlopen` may be missed.
2.  Compatibility cannot be fully guaranteed if binaries are compiled against newer versions of `libc` or `libstdc++`.