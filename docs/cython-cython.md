# Cython: Write Pythonic Code, Get C Performance

Cython is a powerful optimizing static compiler that transforms your Python code into highly efficient C extensions, seamlessly bridging the gap between Python's ease of use and C's speed. ([See the original repo](https://github.com/cython/cython))

## Key Features

*   **Blazing Fast Performance:** Cython compiles Python code into C/C++, enabling significant speed improvements compared to pure Python, similar to C code performance.
*   **Pythonic Syntax:** Write C extensions using a Python-like syntax, making it easy for Python developers to learn and use.
*   **C/C++ Integration:** Seamlessly call C functions and declare C types, allowing for fine-grained control and optimization.
*   **Ideal for Wrapping C Libraries:** Easily integrate and use existing C libraries within your Python projects.
*   **Wide Compatibility:** Works with all versions of CPython.
*   **Large Community & Ecosystem:** Supported by a large user base with extensive documentation and resources.
*   **Over 70 Million Monthly Downloads:** Cython is a widely adopted solution for performance-critical Python tasks, as seen on PyPI.

## Installation

To install Cython, assuming you have a C compiler, simply run:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.  The original Pyrex program, which Cython is based on, was licensed "free of restrictions".

See `LICENSE.txt <https://github.com/cython/cython/blob/master/LICENSE.txt>`_ for the full license details.

## Contributing

We welcome contributions to the Cython project!  Get started by reviewing the [contribution guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython stands out from other Python compilers like PyPy, Numba, mypyc, Nuitka, and Pythran. It provides:

*   Fast and highly compliant support for Python language features
*   Runtime compatibility with all still-in-use and future versions of CPython
*   "Generate once, compile everywhere" C code generation
*   C compile time adaptation to the target platform and Python version
*   Support for other C-API implementations, including PyPy and Pyston
*   Seamless integration with C/C++ code
*   Broad support for manual optimization and tuning down to the C level
*   A large user base with thousands of libraries, packages, and tools
*   Over two decades of bug fixing and static code optimisations

## Project Resources

*   **Official Website:** [https://cython.org/](https://cython.org/)
*   **Documentation:** [https://docs.cython.org/](https://docs.cython.org/)
*   **GitHub Repository:** [https://github.com/cython/cython](https://github.com/cython/cython)
*   **Wiki:** [https://github.com/cython/cython/wiki](https://github.com/cython/cython/wiki)

## Support Cython

You can support the Cython project via:
*   [Github Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)