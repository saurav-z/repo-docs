# Cython: The Ultimate Python Compiler for Speed and Efficiency

**Cython empowers Python developers to write high-performance C extensions with ease, bridging the gap between Python's elegance and C's raw power.**  [Explore the Cython project on GitHub](https://github.com/cython/cython).

## Key Features of Cython:

*   **Python-to-C/C++ Translation:** Converts Python code into highly optimized C/C++ code for significant performance gains.
*   **C Integration:** Seamlessly integrates with C functions and allows declaration of C types, enabling fine-grained control and optimization.
*   **Ideal for C Library Wrapping:** A perfect solution for wrapping external C libraries and creating fast C modules to accelerate Python execution.
*   **Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython.
*   **Cross-Platform C Code Generation:** "Generate once, compile everywhere" C code generation ensures reproducible performance results and easy testing.
*   **Extensive Manual Tuning:** Offers broad support for manual optimization and tuning down to the C level.
*   **Mature and Widely Used:** Boasts a large user base and over two decades of development, with millions of monthly downloads and thousands of libraries, packages, and tools built on it.

## Installation

To install Cython, simply use pip:

```bash
pip install Cython
```

If you encounter issues, refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for detailed instructions.

## License

Cython is licensed under the permissive **Apache License**.  The original Pyrex program, upon which Cython is based, was licensed "free of restrictions".

See the [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) file for more details.

## Contributing

Interested in contributing to the Cython project? Get started with the [contribution guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Comparison with Other Python Compilers

Cython stands out from other Python compilers like PyPy, Numba, mypyc, and Nuitka by offering:

*   **High Compliance:** Comprehensive support for Python language features, including dynamic features and introspection.
*   **Flexibility:** Compile-time adaptation to the target platform and Python version.
*   **C-API Implementation Support:** Support for other C-API implementations, including PyPy and Pyston.
*   **Mature Optimization:** Years of bug fixing and static code optimizations.

## Support the Project

You can support the Cython project through:

*   [GitHub Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)