# Cython: Optimize Python Code with C Speed

**Cython is a powerful optimizing static compiler for both the Python programming language and the extended Cython programming language, enabling you to write C extensions for Python as easily as writing Python itself.** ([Original Repo](https://github.com/cython/cython))

## Key Features

*   **Performance Boost:** Cython translates Python code to highly efficient C/C++ code, significantly speeding up execution.
*   **Seamless Integration:** Easily call C functions and declare C types for fine-grained control and optimization.
*   **C Library Wrapping:** Ideal for wrapping external C libraries, expanding Python's capabilities.
*   **Broad Python Support:** Cython offers fast, efficient, and highly compliant support for almost all Python language features, including dynamic features and introspection.
*   **Cross-Platform Compatibility:** Supports CPython and other C-API implementations like PyPy and Pyston.
*   **Manual Optimization:** Provides broad support for manual optimization and tuning down to the C level.
*   **Mature & Widely Used:** Benefit from over two decades of bug fixes and static code optimizations, supported by a large user base with thousands of libraries, packages, and tools.

## Installation

To install Cython, if you already have a C compiler, simply run:

```bash
pip install Cython
```

Otherwise, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.  See the [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) file for details.

## Contributing

Want to help? Get started with the [CONTRIBUTING guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from other Python Compilers

Cython has evolved to outperform many other Python compilers, including the following:

*   **PyPy:**  Offers JIT compilation with runtime optimizations, but has non-CPython runtime and limited compatibility.
*   **Numba:** JIT compiler for a subset of the language, primarily for numerical code. The main limitation is it's limited language support.
*   **Pythran:** A static Python-to-C++ compiler focused on numerical computation.
*   **mypyc:**  A static Python-to-C compiler based on mypy, which provides good support for language features and PEP-484 typing, but lacks low-level optimization.
*   **Nuitka:**  A static Python-to-C compiler that supports application linking, but no support for low-level optimisations and typing

## Support the Project

*   **GitHub Sponsors:** [Sponsor Cython](https://github.com/users/scoder/sponsorship)
*   **Tidelift:** [Subscribe to Cython](https://tidelift.com/subscription/pkg/pypi-cython)