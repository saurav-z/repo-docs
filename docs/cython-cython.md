# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write high-performance C extensions for Python as easily as you write Python itself, unlocking significant speed improvements for your code.** [Learn more about Cython](https://github.com/cython/cython).

## Key Features:

*   **Python to C/C++ Compilation:** Cython translates your Python code into efficient C/C++ code, optimizing performance.
*   **Seamless C Integration:** Easily call C functions and declare C types within your Python code for fine-grained control and optimization.
*   **Ideal for Wrapping C Libraries:** Simplifies the process of integrating existing C libraries into your Python projects.
*   **High-Performance C Modules:** Create fast C modules to accelerate the execution of your Python code.
*   **Large User Base & Mature Development:** Benefit from over two decades of development, a vast community, and thousands of libraries and tools.
*   **Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython and other C-API implementations.

## Installation

If you have a C compiler already, installation is simple:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive [Apache License](https://github.com/cython/cython/blob/master/LICENSE.txt).

## Contributing

We welcome contributions! Find out how you can [get started contributing to the Cython project](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from Other Python Compilers

Cython, originally based on Pyrex, has evolved to provide several advantages over other Python compilers:

*   **Fast & Compliant:** Supports almost all Python language features with high compliance.
*   **Reproducible Performance:** Generates C code for reproducible performance results and testing.
*   **Platform Adaptation:** Adapts to the target platform and Python version at compile time.
*   **Manual Optimization:** Offers broad support for manual optimization and tuning down to the C level.

Other relevant projects include PyPy, Numba, Pythran, mypyc, and Nuitka. Cython provides advantages over each of these in various ways.

## Support Cython

Support the Cython project via:

*   [GitHub Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

```