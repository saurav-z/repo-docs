# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python with Python's familiar syntax, optimizing your code for speed and efficiency.**

[View the original repository on GitHub](https://github.com/cython/cython)

## Key Features:

*   **Python-like Syntax for C Extensions:** Cython translates Python code into C/C++ code, making it easy to write high-performance extensions.
*   **Seamless C/C++ Integration:** Call C functions and declare C types directly within your Cython code for fine-grained control and optimization.
*   **Broad Optimization Capabilities:** Achieve significant speed improvements by leveraging Cython's features for manual tuning and efficient C code generation.
*   **Ideal for Wrapping C Libraries:** Easily integrate and utilize existing C libraries within your Python projects.
*   **Fast C Modules:** Accelerate the execution of Python code by creating high-speed C modules.
*   **Highly Compliant:** Supports nearly all Python language features, including dynamic features and introspection.
*   **CPython Compatibility:** Full runtime compatibility with CPython.
*   **"Generate Once, Compile Everywhere" Code:** Enables reproducible performance results and thorough testing.
*   **Active Community and Extensive Adoption:** A large user base with thousands of libraries, packages, and tools actively using Cython.
*   **Mature and Stable:** Benefiting from over two decades of bug fixing and static code optimizations.

## Installation

If you have a C compiler installed, install Cython with:

```bash
pip install Cython
```

For more detailed installation instructions, refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.

See `LICENSE.txt <https://github.com/cython/cython/blob/master/LICENSE.txt>`_.

## Contributing

Interested in contributing to the Cython project? Find out how to get started with [these contributing guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from Other Python Compilers

Cython, originating in the early 2000s, has outlasted several other attempts at creating static compilers for Python. Key differences and comparisons with other projects include:

*   **PyPy:** A Python implementation with a JIT compiler. (More information in the original README)
*   **Numba:** A Python extension with a JIT compiler targeting numerical code. (More information in the original README)
*   **Pythran:** A static Python-to-C++ extension compiler mainly for numerical computation. (More information in the original README)
*   **mypyc:** A static Python-to-C extension compiler based on mypy. (More information in the original README)
*   **Nuitka:** A static Python-to-C extension compiler. (More information in the original README)

## History

Cython was originally based on [Pyrex](https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/) by Greg Ewing.