# Cython: The Ultimate Optimizer for Python

**Cython empowers Python developers to write C extensions with Python's ease, unlocking blazing-fast performance.**

[View the Cython Repository on GitHub](https://github.com/cython/cython)

## Key Features:

*   **Performance Enhancement:** Cython translates Python code to C/C++ code, significantly accelerating execution speed.
*   **C/C++ Integration:** Seamlessly integrate with C functions and declare C types, enabling fine-grained optimization.
*   **Wrapping External Libraries:** Easily wrap and utilize existing C libraries within your Python projects.
*   **C Module Creation:** Develop high-performance C modules to supercharge your Python applications.
*   **Cross-Platform Compatibility:** "Generate once, compile everywhere" C code, ensuring reproducible performance results and facilitating testing across different platforms and Python versions.
*   **Mature and Stable:** Benefit from over two decades of bug fixing, static code optimizations, and a large, active user base.
*   **Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython.

## Installation

If you have a C compiler, simply run:

```bash
pip install Cython
```

Otherwise, refer to the [installation documentation](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive [Apache License](https://github.com/cython/cython/blob/master/LICENSE.txt).

## Contributing

Interested in contributing? Get started with the [contribution guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Alternatives and Comparison

Cython stands out among other Python compilers like PyPy, Numba, Pythran, mypyc, and Nuitka.  It provides a unique balance of performance, compatibility, and ease of use:

| Feature                     | Cython                                        |
|-----------------------------|-----------------------------------------------|
| **Language Support**        |  Almost all Python features                  |
| **CPython Compatibility**  |  Full runtime compatibility                  |
| **C/C++ Integration**       |  Seamless integration                      |
| **Optimization Control**   | Broad support for manual optimization     |
| **Maturity**                | Over two decades of development               |

## History and Origins

Cython evolved from Pyrex, created by Greg Ewing.  Pyrex was a language for writing Python extension modules with no restrictions, which Cython builds upon.

## Support the Project

Support the Cython project via
`Github Sponsors <https://github.com/users/scoder/sponsorship>`_ or
`Tidelift <https://tidelift.com/subscription/pkg/pypi-cython>`_.