# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as you write Python itself, boosting performance and integrating seamlessly with C/C++ code.**  Explore the power of Cython for optimized Python development.

[Visit the Cython GitHub Repository](https://github.com/cython/cython)

## Key Features of Cython

*   **Compile Python to C/C++:** Translate Python code into efficient C/C++ code for significant performance gains.
*   **C/C++ Integration:** Seamlessly call C functions and declare C types within your Python code.
*   **Ideal for C Library Wrapping:** Easily create Python bindings for existing C libraries.
*   **Fast C Modules:** Speed up critical sections of your Python code with highly optimized C modules.
*   **Full CPython Compatibility:** Enjoy complete runtime compatibility with all current and future versions of CPython.
*   **Cross-Platform C Code Generation:** Generate C code that compiles and runs on various platforms.
*   **Mature and Widely Used:**  Benefiting from almost two decades of development and a large user community, Cython boasts over 60 million monthly downloads on PyPI.
*   **Extensive Optimization Options:**  Fine-tune performance with manual optimization and C-level control.
*   **Supports other C-API implementations:** including PyPy and Pyston

## Installation

To install Cython, assuming you have a C compiler, simply run:

```bash
pip install Cython
```

For more detailed installation instructions, refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**. See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for details.

## Contributing

Interested in contributing to the Cython project? Get started with the [contributing guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences Compared to Other Python Compilers

Cython has outlived most other attempts at static compilers for Python.  Here is a quick overview of Cython and other projects in the Python ecosystem, that offer similar functionality:

*   [PyPy](https://www.pypy.org/): Python implementation with a JIT compiler.
*   [Numba](http://numba.pydata.org/):  JIT compiler for a subset of the language, based on LLVM.
*   [Pythran](https://pythran.readthedocs.io/): Static Python-to-C++ extension compiler for numerical computation.
*   [mypyc](https://mypyc.readthedocs.io/): Static Python-to-C extension compiler, based on mypy.
*   [Nuitka](https://nuitka.net/): Static Python-to-C extension compiler.

## Support the Project

Show your support for Cython! You can contribute via:

*   [GitHub Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

## Get the Full Source History

To retrieve the complete source history from a downloaded Cython source archive, run:

```bash
make repo