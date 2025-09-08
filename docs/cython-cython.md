# Cython: Optimize Python with C Speed

**Cython is a powerful optimizing compiler that transforms Python code into C/C++ code, enabling you to achieve C-level performance while writing in a familiar Python syntax.** Discover the power of Cython and supercharge your Python projects!

[View the Cython Repository on GitHub](https://github.com/cython/cython)

## Key Features:

*   **Python to C/C++ Compilation:** Converts Python code, along with C function calls and C type declarations, into highly efficient C/C++ code.
*   **Seamless C/C++ Integration:** Easily call C functions and integrate with existing C/C++ libraries.
*   **Fine-Grained Optimization:** Provides tools for manual tuning, allowing you to generate highly optimized C code.
*   **Ideal for Performance-Critical Tasks:** Perfect for creating fast C modules and wrapping external C libraries.
*   **Widely Used and Trusted:** Boasts over 70 million monthly downloads on PyPI, a testament to its reliability and performance.
*   **CPython Compatibility:** Ensures full runtime compatibility with current and future CPython versions.
*   **Cross-Platform Support:** Generates C code that can be compiled on various platforms.
*   **Mature and Stable:** Benefit from over two decades of bug fixes and static code optimizations.

## Installation

To install Cython, simply run:

```bash
pip install Cython
```

If you don't have a C compiler, consult the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for more detailed instructions.

## License

Cython is licensed under the permissive **Apache License**.

See the [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) file for more information.

## Contributing

Interested in contributing to the Cython project? Get started with the help available at [CONTRIBUTING.rst](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython offers unique advantages compared to other Python compilers like PyPy, Numba, mypyc, and Nuitka. Cython provides:

*   Highly compatible support for almost all Python language features.
*   "Generate once, compile everywhere" C code generation.
*   C compile time adaptation to the target platform and Python version.
*   Seamless integration with C/C++ code.
*   Broad support for manual optimization and tuning.

## Supporting Cython

You can support the Cython project through:

*   [GitHub Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

## The Pyrex Legacy

Cython builds upon the foundation of the original Pyrex project. See the `original Pyrex source <https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/>`_.