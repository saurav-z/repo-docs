# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as you write Python itself, unlocking significant performance gains.**  [Learn more about Cython](https://github.com/cython/cython).

## Key Features of Cython

*   **Python to C/C++ Compilation:** Translates Python code into highly optimized C/C++ code.
*   **C/C++ Integration:** Seamlessly calls C functions and declares C types for efficient code generation.
*   **Ideal for:** Wrapping external C libraries and accelerating the execution of Python code.
*   **Performance:** Can generate efficient C code for faster execution of Python code.
*   **Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython.
*   **Optimization:** Broad support for manual optimization and tuning down to the C level.
*   **Large Community:** A large user base with thousands of libraries, packages, and tools.

## Installation

To install Cython, if you have a C compiler, simply run:

```bash
pip install Cython
```

Otherwise, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for more detailed instructions.

## License

Cython is licensed under the permissive **Apache License**.

See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for details.

## Contributing

Interested in contributing to the Cython project?  Find resources to get started at [CONTRIBUTING.rst](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences Compared to Other Python Compilers

Cython distinguishes itself from other Python compilers like PyPy, Numba, Pythran, mypyc, and Nuitka by offering:

*   Fast, efficient, and highly compliant support for almost all Python language features, including dynamic features and introspection
*   "Generate once, compile everywhere" C code generation that allows for reproducible performance results and testing
*   C compile-time adaptation to the target platform and Python version
*   Support for other C-API implementations, including PyPy and Pyston
*   Seamless integration with C/C++ code

## Resources

*   **Official Website:** [https://cython.org/](https://cython.org/)
*   **Documentation:** [https://docs.cython.org/](https://docs.cython.org/)
*   **GitHub Repository:** [https://github.com/cython/cython](https://github.com/cython/cython)
*   **Wiki:** [https://github.com/cython/cython/wiki](https://github.com/cython/cython/wiki)

## Support the Project

*   **GitHub Sponsors:** [https://github.com/users/scoder/sponsorship](https://github.com/users/scoder/sponsorship)
*   **Tidelift:** [https://tidelift.com/subscription/pkg/pypi-cython](https://tidelift.com/subscription/pkg/pypi-cython)

## Get the Full Source History

To obtain the complete source history from a downloaded source archive, ensure you have `git` installed, navigate to the Cython source distribution's base directory, and execute:

```bash
make repo
```