# auditwheel: Ensure Linux Wheel Compatibility for Python Packages

**auditwheel** is a powerful command-line tool designed to audit and repair Python wheel packages, ensuring compatibility with various Linux distributions, including those adhering to the PEP 600 manylinux standards. [See the original repository](https://github.com/pypa/auditwheel) for more information.

## Key Features

*   **Auditing:** Analyzes Python wheel packages to identify external shared library dependencies and potential compatibility issues.
*   **Repairing:** Modifies wheels to include necessary shared libraries, resolving dependency conflicts and ensuring compatibility with older Linux distributions.
*   **Manylinux Compliance:**  Facilitates creation of wheels that conform to manylinux1, manylinux2010, manylinux2014 and manylinux_x_y standards for broader Linux distribution support.
*   **RPATH Modification:** Automatically adjusts `RPATH` entries to ensure that bundled libraries are correctly located at runtime.

## Overview

auditwheel helps you create Python wheel packages for Linux (containing pre-compiled binary extensions) that are compatible with a wide variety of Linux distributions, consistent with the `PEP 600 manylinux_x_y <https://www.python.org/dev/peps/pep-0600/>`_, `PEP 513 manylinux1 <https://www.python.org/dev/peps/pep-0513/>`_, `PEP 571 manylinux2010 <https://www.python.org/dev/peps/pep-0571/>`_ and `PEP 599 manylinux2014 <https://www.python.org/dev/peps/pep-0599/>`_ platform tags.

**Commands:**

*   `auditwheel show`: Shows external shared libraries that the wheel depends on and checks the extension modules for the use of versioned symbols that exceed the `manylinux` ABI.
*   `auditwheel repair`: Copies external shared libraries into the wheel and modifies the appropriate `RPATH` entries such that these libraries will be picked up at runtime.

## Requirements

*   **OS:** Linux
*   **Python:** 3.9+
*   **patchelf:** 0.14+

## Installation

Install auditwheel using pip:

```bash
pip3 install auditwheel
```

## Examples

**Inspecting a wheel:**

```bash
$ auditwheel show cffi-1.5.0-cp35-cp35m-linux_x86_64.whl
```

**Repairing a wheel:**

```bash
$ auditwheel repair cffi-1.5.2-cp35-cp35m-linux_x86_64.whl
```

## Limitations

*   Doesn't detect dependencies loaded dynamically via `ctypes`, `cffi`, or `dlopen`.
*   Cannot "fix" binaries compiled against excessively new versions of `libc` or `libstdc++`.

## Testing

Run tests with `nox`. Integration tests may require a running Docker daemon.

## Code of Conduct

Everyone interacting in the `auditwheel` project's codebases, issue trackers, chat rooms, and mailing lists is expected to follow the `PSF Code of Conduct <https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md>`.