# Cython: The Ultimate Python Compiler for Speed and Efficiency

**Cython empowers developers to write C extensions for Python with ease, enabling significant performance gains.**  Leveraging a Python-like syntax, Cython translates Python code into highly optimized C/C++ code, making it the ideal choice for speeding up Python applications, integrating with existing C/C++ libraries, and building high-performance modules.  Learn more and contribute at the [original Cython repository](https://github.com/cython/cython).

## Key Features:

*   **Seamless Python Integration:**  Write C extensions using a Python-like syntax, making it easy for Python developers to get started.
*   **C/C++ Interoperability:**  Call C functions and declare C types directly within your Python code for fine-grained control and optimization.
*   **Performance Boost:** Generate very efficient C code from Cython code, dramatically accelerating your Python applications.
*   **External Library Wrapping:**  Easily integrate and wrap existing C and C++ libraries for use in Python.
*   **Cross-Platform Compatibility:**  Generate C code that can be compiled and run on various platforms and Python versions.
*   **Mature and Widely Used:**  Benefit from a stable, well-documented project with a large community and over 70 million downloads per month.
*   **Manual Optimization Support:**  Offers broad support for manual optimization and tuning down to the C level.

## Installation

If you have a C compiler installed, install Cython with:

```bash
pip install Cython
```

Otherwise, follow the detailed installation instructions in the [Cython documentation](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.  See the full license details in [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt).

## Contributing

Help improve Cython! Find guidance for contributions in the [CONTRIBUTING.rst](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst) file.

## Cython Compared to Other Python Compilers

Cython distinguishes itself from other Python compilers, such as PyPy, Numba, mypyc, and Nuitka, by providing:

*   **Comprehensive Python Support:** Maintains fast, efficient, and highly compliant support for nearly all Python language features, including dynamic features and introspection.
*   **CPython Compatibility:** Full runtime compatibility with all existing and future versions of CPython.
*   **Reproducible Performance:** "Generate once, compile everywhere" C code generation, allowing reproducible performance results and robust testing.
*   **Platform Adaptability:** C compile-time adaptation to the target platform and Python version.
*   **C-API Implementation Support:** Support for other C-API implementations, including PyPy and Pyston.
*   **Seamless C/C++ Integration:** Effortless integration with existing C/C++ code.
*   **Manual Optimization:** Extensive options for manual optimization and tuning down to the C level.
*   **Large User Community:** A significant user base, supported by thousands of libraries, packages, and tools.
*   **Long-Term Stability:** Over two decades of experience in bug fixing and static code optimization.

## Project History

Cython is based on the original Pyrex project by Greg Ewing. Pyrex, written "free of restrictions", provided the foundation for Cython's evolution into a powerful and versatile Python compiler.

## Support Cython

Support the project through [Github Sponsors](https://github.com/users/scoder/sponsorship) or [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython).