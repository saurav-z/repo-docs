# auditwheel: Ensure Linux Wheel Compatibility

**auditwheel is your go-to command-line tool for creating and auditing Python wheel packages, ensuring compatibility with various Linux distributions.** Learn more on the [original repo](https://github.com/pypa/auditwheel).

## Key Features

*   **Auditing:**
    *   Identifies external shared library dependencies within your wheel packages.
    *   Checks for versioned symbols that exceed `manylinux` ABI compatibility.
*   **Repairing:**
    *   Copies external shared libraries into your wheel package.
    *   Automatically modifies `RPATH` entries for correct runtime linking.
    *   Transforms the wheel's platform tags for improved compatibility (e.g. `linux_x86_64` to `manylinux1_x86_64`).
*   **Compatibility:**
    *   Supports `PEP 600 manylinux_x_y`, `PEP 513 manylinux1`, `PEP 571 manylinux2010`, and `PEP 599 manylinux2014` standards.
*   **Easy Installation:**
    *   Install using pip: `pip3 install auditwheel`

## How to Use

### Inspecting a Wheel

```bash
auditwheel show my_package-1.0.0-cp37-cp37m-linux_x86_64.whl
```

This will show external shared libraries and any compatibility issues.

### Repairing a Wheel

```bash
auditwheel repair my_package-1.0.0-cp37-cp37m-linux_x86_64.whl
```

This will modify the wheel, making it compatible with a wider range of Linux distributions.

## Requirements

*   **OS:** Linux
*   **Python:** 3.9+
*   **patchelf:** 0.14+ (required)

## Limitations

*   Doesn't detect dependencies loaded dynamically via `ctypes`, `cffi` or `dlopen`.
*   Cannot "fix" binaries compiled against excessively new versions of `libc` or `libstdc++`. It is recommended to build wheels in a `manylinux` Docker image to ensure compatibility.

## Testing

Run tests with `nox`. Some integration tests require a running Docker daemon. Update Docker images manually with the commands provided in the original README if needed.

## Code of Conduct

This project adheres to the [PSF Code of Conduct](https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md).