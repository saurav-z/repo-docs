# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write blazing-fast C extensions for Python as easily as you write Python itself.**  Learn more at the [Cython GitHub repository](https://github.com/cython/cython).

## Key Features

*   **Performance Boost:** Translate Python code into highly efficient C/C++ code, significantly speeding up your applications.
*   **C/C++ Integration:** Seamlessly call C functions and declare C types within your Python code for fine-grained control and optimization.
*   **Versatile Applications:** Ideal for wrapping external C libraries and creating fast C modules to accelerate Python code execution.
*   **Broad Compatibility:** Fully compatible with CPython and offers support for other C-API implementations, including PyPy and Pyston.
*   **Mature & Reliable:** Benefit from two decades of development, bug fixes, and static code optimizations.
*   **Large Community:** Leverage a vast user base with thousands of libraries, packages, and tools built with Cython.

## Installation

If you have a C compiler, install Cython with:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) in the documentation.

## License

Cython is licensed under the permissive **Apache License**.

See the full license in `LICENSE.txt <https://github.com/cython/cython/blob/master/LICENSE.txt>`.

## Contributing

Contribute to the Cython project! Find helpful resources to get started in the [CONTRIBUTING guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from Other Python Compilers

Cython distinguishes itself from other Python compilers through:

*   **High language compliance** and support for dynamic features and introspection.
*   **"Generate once, compile everywhere"** C code generation for reproducible performance and testing.
*   **C compile-time adaptation** to the target platform and Python version.
*   **Seamless integration with C/C++ code.**
*   **Extensive support for manual optimization** and tuning at the C level.

The following are similar projects:

*   `PyPy <https://www.pypy.org/>`_
*   `Numba <http://numba.pydata.org/>`_
*   `Pythran <https://pythran.readthedocs.io/>`_
*   `mypyc <https://mypyc.readthedocs.io/>`_
*   `Nuitka <https://nuitka.net/>`_