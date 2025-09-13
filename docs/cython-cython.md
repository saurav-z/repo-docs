# Cython: Write C Extensions for Python with Ease

**Cython empowers Python developers to write high-performance C extensions, making it easy to bridge the gap between Python and C/C++ for significant speed gains.**  [Explore the Cython project on GitHub](https://github.com/cython/cython).

## Key Features of Cython:

*   **Python-like Syntax:** Cython allows you to write C extensions using a Python-like syntax, making it easier to learn and use compared to writing C directly.
*   **C/C++ Integration:** Seamlessly integrate with C and C++ code, allowing you to call C functions and declare C types for optimized performance.
*   **Optimizing Compiler:** Cython translates Python code into optimized C/C++ code, providing significant speed improvements for computationally intensive tasks.
*   **Fine-Grained Control:**  Offers the ability to tune performance with manual optimizations, enabling efficient C code generation.
*   **Ideal for Wrapping C Libraries:** Easily wrap existing C libraries for use in Python, extending the capabilities of your Python projects.
*   **Cross-Platform Compatibility:** "Generate once, compile everywhere" C code generation that allows for reproducible performance results and testing.
*   **CPython Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython.
*   **Large Community & Ecosystem:** Supported by a large user base, with thousands of libraries, packages, and tools.

## Installation

If you have a C compiler, install Cython with:

```bash
pip install Cython
```

For more detailed instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## Support the Project

Show your support for Cython!

*   [GitHub Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

## License

Cython is licensed under the permissive [Apache License](https://github.com/cython/cython/blob/master/LICENSE.txt).

## Contributing

Interested in contributing to the Cython project? Get started with the [contributing guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from Other Python Compilers

Cython stands out among other Python compilers with its focus on:

*   Fast, efficient, and highly compliant support for Python language features, including dynamic features and introspection.
*   Broad support for manual optimization and tuning down to the C level.
*   Two decades of bug fixing and static code optimizations.

Key projects offering similar functionality include:

*   **PyPy:** A Python implementation with a JIT compiler.
*   **Numba:** A Python extension with a JIT compiler for numerical code.
*   **Pythran:** A static Python-to-C++ extension compiler, often used as a NumPy backend in Cython.
*   **mypyc:** A static Python-to-C extension compiler, based on mypy type annotations.
*   **Nuitka:** A static Python-to-C extension compiler.

## Additional Resources

*   [Official Website](https://cython.org/)
*   [Documentation](https://docs.cython.org/)
*   [GitHub Repository](https://github.com/cython/cython)
*   [Wiki](https://github.com/cython/cython/wiki)