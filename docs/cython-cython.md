# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as you write Python itself, significantly boosting performance.**

[Visit the original Cython repository on GitHub](https://github.com/cython/cython)

## Key Features

*   **Python-to-C/C++ Compilation:** Translates Python code into highly efficient C/C++ code.
*   **C/C++ Integration:** Seamlessly calls C functions and declares C types within your Python code.
*   **Fine-Grained Tuning:** Offers precise control over code optimization for maximum performance.
*   **Ideal for Wrapping C Libraries:** Simplifies the integration of external C libraries.
*   **Fast C Modules:** Accelerates Python code execution through optimized C modules.
*   **Cross-Platform Compatibility:** "Generate once, compile everywhere" C code generation for reproducible performance.
*   **Extensive Python Support:** Comprehensive support for Python features, including dynamic features and introspection.
*   **CPython Compatibility:** Full runtime compatibility with current and future CPython versions.
*   **Mature and Widely Used:** Benefiting from over two decades of development, Cython boasts a large user base and community.

## Installation

If you have a C compiler already installed, installing Cython is as easy as running:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.

See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for the full license details.

## Contributing

Interested in contributing to the Cython project? Find information on how to get started in the [CONTRIBUTING guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from Other Python Compilers

Cython stands out among Python compilers due to its:

*   **Performance:** Cython provides fast, efficient, and highly compliant support for almost all Python language features.
*   **Compatibility:**  Full runtime compatibility with CPython.
*   **Optimization:** Broad support for manual optimization down to the C level.
*   **Maturity:** Backed by a large user base and a long history of development.

Other Python compilers that you may find:

*   [PyPy](https://www.pypy.org/): JIT compiler with runtime optimizations.
*   [Numba](http://numba.pydata.org/): JIT compiler for a subset of the language, mostly targeting numerical code.
*   [Pythran](https://pythran.readthedocs.io/): Python-to-C++ extension compiler, best used as a backend for NumPy code in Cython.
*   [mypyc](https://mypyc.readthedocs.io/): Python-to-C extension compiler, makes use of PEP-484 type annotations to optimize code for static types.
*   [Nuitka](https://nuitka.net/): Python-to-C extension compiler, supporting static application linking.

## History

Cython's origins are based on the [Pyrex](https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/) project, originally developed by Greg Ewing.