# Cython: Write C Extensions for Python with Ease

**Cython empowers developers to write C extensions for Python as easily as they write Python itself, enabling significant performance improvements.**  Learn more about this powerful tool and how it can accelerate your Python projects. [Visit the Cython GitHub Repository](https://github.com/cython/cython) to get started.

## Key Features of Cython:

*   **Python-to-C/C++ Compilation:** Translates Python code into highly efficient C/C++ code.
*   **Seamless C Integration:**  Allows direct calls to C functions and declaration of C types for fine-grained control.
*   **Performance Optimization:** Offers broad, manual tuning capabilities to generate very efficient C code.
*   **Ideal for Wrapping C Libraries:** Simplifies the process of integrating external C libraries into your Python projects.
*   **Fast C Modules:** Enables the creation of fast C modules to speed up Python code execution.
*   **Full CPython Compatibility:** Ensures runtime compatibility with all current and future versions of CPython.
*   **Widely Adopted:** Used by thousands of libraries, packages, and tools, with over 70 million downloads per month on PyPI.

## Installation

To install Cython, if you have a C compiler, simply run:

```bash
pip install Cython
```

For detailed installation instructions, please see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.

See `LICENSE.txt <https://github.com/cython/cython/blob/master/LICENSE.txt>`_.

## Contributing

Interested in contributing to the Cython project? Get started with our [contribution guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from Other Python Compilers

Cython has outlived many attempts at producing static compilers for the Python language. Similar projects include:

*   [PyPy](https://www.pypy.org/)
*   [Numba](http://numba.pydata.org/)
*   [Pythran](https://pythran.readthedocs.io/)
*   [mypyc](https://mypyc.readthedocs.io/)
*   [Nuitka](https://nuitka.net/)

Cython's advantages include:

*   Fast, highly compliant support for almost all Python features.
*   "Generate once, compile everywhere" C code generation for reproducible results.
*   Compile-time adaptation to the target platform and Python version.
*   Seamless integration with C/C++ code.
*   Broad support for manual optimization.
*   A large and active user base.

##  Support the Cython Project

You can support the Cython project via
`Github Sponsors <https://github.com/users/scoder/sponsorship>`_ or
`Tidelift <https://tidelift.com/subscription/pkg/pypi-cython>`_.