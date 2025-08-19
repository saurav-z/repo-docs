# Cython: Write C Extensions for Python with Ease

**Cython empowers developers to write high-performance C extensions for Python as easily as writing Python itself.**  For the original repository, see [here](https://github.com/cython/cython).

## Key Features

*   **Python-to-C/C++ Compilation:** Translates Python code into efficient C/C++ code.
*   **C/C++ Integration:** Seamlessly calls C functions and declares C types for fine-grained control and optimization.
*   **Performance Enhancement:** Ideal for speeding up Python code and wrapping external C libraries.
*   **Broad Language Support:**  Offers fast, efficient, and highly compliant support for almost all Python language features, including dynamic features and introspection.
*   **CPython Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython.
*   **Platform Adaptability:** C compile-time adaptation to the target platform and Python version.
*   **C-API Support:**  Supports other C-API implementations, including PyPy and Pyston.
*   **Optimization:** Broad support for manual optimization and tuning down to the C level.
*   **Mature & Widely Used:**  A large user base with thousands of libraries, packages, and tools and two decades of development.

## Installation

To install Cython, simply run:

```bash
pip install Cython
```

If you don't have a C compiler, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for instructions.

## License

Cython is licensed under the permissive [Apache License](https://github.com/cython/cython/blob/master/LICENSE.txt).

## Contributing

Want to contribute to Cython? Get started with the [contributing guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython stands out as a powerful tool for performance optimization in Python. Here's how it compares to other options:

*   **PyPy:** Python implementation with a JIT compiler.
*   **Numba:** JIT compiler for a subset of the language, primarily for numerical code using NumPy.
*   **Pythran:** Static Python-to-C++ compiler, focused on numerical computation, often used as a backend for NumPy code within Cython.
*   **mypyc:** Static Python-to-C extension compiler, making use of PEP-484 type annotations for optimization.
*   **Nuitka:** Static Python-to-C extension compiler.

**Cython's Advantages:**

*   Fast and efficient implementation of nearly all Python features.
*   Full runtime compatibility with all CPython versions.
*   "Generate once, compile everywhere" C code generation.
*   Seamless integration with C/C++ code.
*   Manual optimization capabilities.

## Support Cython

You can **support the Cython project** via
[Github Sponsors](https://github.com/users/scoder/sponsorship) or
[Tidelift](https://tidelift.com/subscription/pkg/pypi-cython).

## Additional Resources

*   [Official Website](https://cython.org/)
*   [Documentation](https://docs.cython.org/)
*   [GitHub Repository](https://github.com/cython/cython)
*   [Wiki](https://github.com/cython/cython/wiki)

## About Pyrex

Cython was originally based on [Pyrex](https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/) by Greg Ewing. Pyrex was "free of restrictions" to be used and modified.