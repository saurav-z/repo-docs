# auditwheel: Ensure Linux Wheels are Compatible with PEP 600, PEP 513, PEP 571 and PEP 599

**auditwheel** is a powerful command-line tool that simplifies creating and auditing Python wheel packages for Linux, ensuring they are compatible with various Linux distributions.

**[Visit the original repository on GitHub](https://github.com/pypa/auditwheel)**

## Key Features

*   **Auditing and Analysis:**
    *   Inspects Python wheel packages to identify external shared library dependencies.
    *   Checks for versioned symbols that may exceed manylinux ABI compatibility.
    *   Provides insights into wheel compatibility with different Linux distributions.
*   **Wheel Repair:**
    *   Copies external shared libraries into the wheel itself.
    *   Modifies `RPATH` entries to ensure libraries are found at runtime, resolving dependency issues.
    *   Facilitates creation of manylinux wheels without modifying the build system.
*   **Platform Tagging:**
    *   Helps ensure compliance with `PEP 600 manylinux_x_y`, `PEP 513 manylinux1`, `PEP 571 manylinux2010`, and `PEP 599 manylinux2014` standards.
*   **Easy Installation:**
    *   Install with pip: `pip3 install auditwheel`
*   **Requirements:**
    *   OS: Linux
    *   Python: 3.9+
    *   `patchelf`: 0.14+

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
*   Cannot fix binaries compiled against too-recent versions of `libc` or `libstdc++`.
*   Requires running on an ELF-based system.

## Testing

The tests can be run with `nox`.
Some tests require a running and accessible Docker daemon.