# Cython: Write C Extensions for Python with Ease

**Cython is a powerful optimizing compiler that bridges the gap between Python and C, enabling you to write high-performance extensions as easily as you write Python code.**

[View the Cython Repository on GitHub](https://github.com/cython/cython)

## Key Features

*   **Python-to-C Translation:** Cython translates Python code, allowing you to write C extensions directly using a Python-like syntax.
*   **C/C++ Integration:** Seamlessly call C functions and declare C types for fine-grained control and optimization.
*   **Performance Boost:**  Generate highly efficient C code from Cython, significantly speeding up your Python applications, especially for computationally intensive tasks.
*   **Wrapping External C Libraries:** Ideal for integrating and wrapping existing C libraries, expanding the capabilities of your Python projects.
*   **Compatibility:** Full runtime compatibility with CPython and support for other C-API implementations including PyPy and Pyston.
*   **Mature and Widely Used:** A mature project with a large user base, thousands of libraries, packages, and tools, and over 70 million monthly downloads on PyPI.
*   **Broad Optimization Support:**  Offers extensive manual optimization and tuning, allowing you to optimize at the C level.

## Installation

If you have a C compiler installed, you can install Cython with:

```bash
pip install Cython
```

Otherwise, refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) in the documentation.

## License

Cython is licensed under the permissive **Apache License**.
See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for details.

## Contributing

Interested in contributing?  Find help to get started in the [CONTRIBUTING.rst](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst) file.

## Cython vs. Other Python Compilers

Cython stands out from other Python compilers due to its comprehensive support for Python features and its ability to generate highly optimized C code.  Here's a comparison:

*   **PyPy:** Offers JIT compilation, but is a different runtime.
*   **Numba:** Focuses on numerical code using LLVM.
*   **Pythran:**  A static Python-to-C++ compiler for numerical computation.
*   **mypyc:**  A static Python-to-C compiler leveraging type annotations.
*   **Nuitka:**  A static Python-to-C compiler.

Cython provides:

*   Excellent support for Python features
*   Runtime compatibility with future versions of CPython
*   Platform adaptation at compile time
*   Seamless integration with C/C++ code
*   A large user base and extensive bug fixing.

## Origin: Based on Pyrex

Cython evolved from [Pyrex](https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/) by Greg Ewing.  Pyrex, the foundation of Cython, was released under a "free of restrictions" license.