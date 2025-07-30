# Cython: Compile Python to C for Blazing Fast Performance

Cython is a powerful optimising Python compiler that allows you to write C extensions as easily as Python itself, significantly accelerating your code.  Learn more and contribute on the original [Cython GitHub repository](https://github.com/cython/cython).

## Key Features:

*   **Python-to-C Compilation:** Translates Python code into highly efficient C/C++ code.
*   **C/C++ Integration:** Seamlessly calls C functions and declares C types, enabling fine-grained optimization.
*   **Performance Boost:** Ideal for wrapping C libraries and speeding up computationally intensive Python modules.
*   **Full CPython Compatibility:**  Works with all current and future CPython versions.
*   **Broad Optimization Support:** Offers extensive features for manual optimization down to the C level.
*   **Large Ecosystem:**  Supported by a large user base, with thousands of libraries and tools.
*   **Mature and Stable:** Benefit from over two decades of bug fixes and static code optimizations.

## Installation

If you have a C compiler installed, you can easily install Cython using:

```bash
pip install Cython
```

Otherwise, consult the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for detailed instructions.

## License

Cython is licensed under the permissive **Apache License**.  The original Pyrex program, on which Cython is based, was "free of restrictions."

See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for more details.

## Contributing

Interested in contributing to the Cython project? Get started with the [contributing guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from Other Python Compilers

Cython distinguishes itself from other Python compilers like PyPy, Numba, Pythran, mypyc, and Nuitka by offering a unique blend of features:

*   **High Compliance and Dynamic Features:** Provides fast, efficient, and highly compliant support for almost all Python language features, including dynamic features and introspection.
*   **CPython Compatibility:** Ensures full runtime compatibility with all still-in-use and future versions of CPython.
*   **Platform Adaptation:** Generates "once, compile everywhere" C code that adapts to the target platform and Python version.
*   **C-API Implementations Support:**  Supports other C-API implementations, including PyPy and Pyston.
*   **C/C++ Integration:** Allows for seamless integration with C/C++ code.
*   **Optimization Capabilities:** Provides broad support for manual optimization and tuning down to the C level.

## About Pyrex (Original Project)

Cython is based on the original [Pyrex](https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/) project by Greg Ewing.