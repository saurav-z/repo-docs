# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as writing Python itself, enabling significant performance gains.**  You can find the original repository [here](https://github.com/cython/cython).

## Key Features of Cython

*   **Python to C/C++ Compilation:** Translates Python code into highly efficient C/C++ code.
*   **C/C++ Integration:** Seamlessly calls C functions and declares C types within your Python code.
*   **Performance Optimization:** Enables substantial speed improvements for computationally intensive tasks.
*   **External Library Wrapping:** Ideal for wrapping external C libraries, making them accessible from Python.
*   **Fast C Modules:** Creates fast C modules that accelerate Python code execution.
*   **Wide Compatibility:** Supports all still-in-use and future versions of CPython.
*   **Reproducible Performance:** Generates C code that allows for reproducible performance results.
*   **C-API Implementations:** Supports other C-API implementations, including PyPy and Pyston.
*   **Manual Optimization:** Broad support for manual optimization and tuning down to the C level.

## Installation

If you have a C compiler, install Cython with:

```bash
pip install Cython
```

Otherwise, refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) in the documentation.

## License

Cython is licensed under the permissive **Apache License**.

See the [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) file for details.

## Contributing

Contribute to the Cython project - get started with this [help document](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Compared to other Python compilers like PyPy, Numba, Pythran, mypyc, and Nuitka, Cython stands out for its:

*   Fast, efficient, and highly compliant support for almost all Python language features, including dynamic features and introspection
*   Full runtime compatibility with all still-in-use and future versions of CPython
*   "Generate once, compile everywhere" C code generation that allows for reproducible performance results and testing
*   C compile time adaptation to the target platform and Python version
*   Seamless integration with C/C++ code
*   Manual optimization and tuning to the C level

## Additional Resources

*   **Official Website:** [https://cython.org/](https://cython.org/)
*   **Documentation:** [https://docs.cython.org/](https://docs.cython.org/)
*   **GitHub Repository:** [https://github.com/cython/cython](https://github.com/cython/cython)
*   **Wiki:** [https://github.com/cython/cython/wiki](https://github.com/cython/cython/wiki)

## Support the Project

You can support the Cython project via [GitHub Sponsors](https://github.com/users/scoder/sponsorship) or [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython).