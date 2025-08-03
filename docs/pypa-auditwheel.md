# auditwheel: Audit and Repair Linux Wheels for Maximum Compatibility

**auditwheel** is your go-to command-line tool for ensuring Python wheel packages containing compiled extensions are compatible across a wide range of Linux distributions.  Learn more and contribute at the [original repository](https://github.com/pypa/auditwheel).

## Key Features

*   **Auditing:** Identifies external shared library dependencies and versioned symbols within your wheel packages, ensuring they meet the `manylinux` standards (PEP 600, PEP 513, PEP 571, and PEP 599).
*   **Repairing:** Copies required external shared libraries into the wheel and modifies `RPATH` entries to ensure they are found at runtime, increasing compatibility without requiring changes to your build process.
*   **Platform Tagging:**  Updates wheel filename tags to reflect manylinux compliance (e.g., `manylinux1_x86_64`).
*   **Easy Installation:** Install with pip: `pip3 install auditwheel`
*   **Broad Compatibility:** Supports ELF-based Linux systems.

## Overview

`auditwheel` helps you create Python wheel packages that are compatible with a broad range of Linux distributions.  It achieves this by:

*   **Inspecting Wheels:**  Using `auditwheel show` to reveal external shared library dependencies and potential compatibility issues, such as the use of versioned symbols exceeding `manylinux` ABI standards.
*   **Repairing Wheels:**  Using `auditwheel repair` to bundle external shared libraries into the wheel and adjust the `RPATH` entries to ensure they're accessible at runtime. This mimics static linking without the need to change your build setup.

## Requirements

*   **Operating System:** Linux
*   **Python:** 3.9+
*   **Dependency:**  `patchelf` (version 0.14 or greater).  Make sure to install [patchelf](https://github.com/NixOS/patchelf) as a prerequisite.

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

*   **Dynamic Loading:** `auditwheel` relies on `DT_NEEDED` information.  Dependencies loaded dynamically via `ctypes`, `cffi`, or `dlopen` might be missed.
*   **glibc/libstdc++ Versioning:**  `auditwheel` cannot fix binaries compiled against overly new versions of `glibc` or `libstdc++`.  Compile on older Linux distributions, such as manylinux Docker images, for best compatibility.

## Testing

The tests can be run using `nox`.

Some integration tests require a running and accessible Docker daemon. To update related docker images, run:

```bash
docker pull python:3.9-slim-bookworm
docker pull quay.io/pypa/manylinux1_x86_64
docker pull quay.io/pypa/manylinux2010_x86_64
docker pull quay.io/pypa/manylinux2014_x86_64
docker pull quay.io/pypa/manylinux_2_28_x86_64
docker pull quay.io/pypa/manylinux_2_34_x86_64
docker pull quay.io/pypa/musllinux_1_2_x86_64
```
You may also remove these images using `docker rmi`.

## Code of Conduct

Please adhere to the [PSF Code of Conduct](https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md) when interacting with the `auditwheel` project.