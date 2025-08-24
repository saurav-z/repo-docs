# Cython: The High-Performance Compiler for Python

**Supercharge your Python code and seamlessly integrate with C/C++ using Cython, the optimizing compiler that brings the speed of C with the ease of Python!**  For the original repository, visit [Cython on GitHub](https://github.com/cython/cython).

## Key Features of Cython

*   **Blazing-Fast Performance:** Translate Python code to highly efficient C/C++ code, unlocking significant speed improvements.
*   **C/C++ Integration:** Easily call C functions and declare C types, enabling seamless interaction with existing C/C++ libraries and low-level optimizations.
*   **Fine-Grained Control:** Manually tune your code with broad to fine-grained control, allowing for optimal C code generation.
*   **Ideal for Extensions:** Perfect for wrapping external C libraries and creating fast C modules to accelerate Python code execution.
*   **Cross-Platform Compatibility:** Cython generates "generate once, compile everywhere" C code, ensuring reproducible performance across different platforms.
*   **Mature and Widely Used:** Benefit from over two decades of development, a large user base, and millions of monthly downloads.

## Installation

If you have a C compiler installed, simply run:

```bash
pip install Cython
```

Otherwise, refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for detailed instructions.

## Licensing

Cython is licensed under the permissive **Apache License**.  The original Pyrex program, upon which Cython is based, was licensed "free of restrictions".

See the [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) file for more details.

## Contributing

Interested in contributing to the Cython project? Get started with this [guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython stands out from other Python compilers due to its:

*   **Fast, efficient, and compliant support** for almost all Python language features, including dynamic features and introspection
*   **Full runtime compatibility** with all still-in-use and future versions of CPython
*   **C compile time adaptation** to the target platform and Python version
*   **Seamless integration** with C/C++ code
*   **Broad support** for manual optimisation and tuning down to the C level

Compared to other Python compilers like PyPy, Numba, mypyc, and Nuitka, Cython offers a unique balance of performance, flexibility, and compatibility.

## History: Based on Pyrex

Cython builds upon the foundation of [Pyrex](https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/), a language for writing Python extension modules, created by Greg Ewing.