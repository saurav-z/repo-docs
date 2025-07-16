# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write blazing-fast C extensions for Python, seamlessly bridging the gap between Python's elegance and C's performance.**  [Explore the Cython repository on GitHub](https://github.com/cython/cython).

## Key Features

*   **Python-to-C/C++ Compilation:** Cython translates Python code into highly optimized C/C++ code, allowing for significant performance gains.
*   **C/C++ Integration:** Easily call C functions and declare C types within your Python code, enabling efficient interaction with existing C libraries.
*   **Simplified Extension Development:** Cython simplifies the process of writing C extensions, making it as straightforward as writing Python itself.
*   **Ideal for Performance-Critical Tasks:** Perfect for accelerating Python code, wrapping external C libraries, and creating high-performance modules.
*   **Mature and Widely Used:** With over 60 million monthly downloads on PyPI, Cython is a proven solution for performance optimization.
*   **CPython Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython
*   **Cross-Platform Support:** "Generate once, compile everywhere" C code generation that allows for reproducible performance results and testing
*   **Broad Optimization:** Support for manual optimization and tuning down to the C level
*   **Seamless C/C++ Integration:** Full support for seamless integration with C/C++ code.
*   **Large User Base:** A large user base with thousands of libraries, packages and tools.

## Installation

If you have a C compiler already set up, install Cython easily with:

```bash
pip install Cython
```

For more detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**. See the [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) file for details.

## Contributing

Interested in contributing to the Cython project?  Get started with these [contribution guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Comparing Cython to Other Python Compilers

Cython has evolved over two decades, outlasting many other attempts to produce static compilers for Python. Here's how Cython stacks up against other Python compilers:

*   **PyPy:** JIT compiler; offers good runtime optimizations, but may have larger resource usage and limited CPython extension compatibility.
*   **Numba:** JIT compiler for a subset of Python; focuses on numerical code, but has limited language support and runtime dependencies.
*   **Pythran:** Static Python-to-C++ compiler for numerical computation; can be used with NumPy and Cython.
*   **mypyc:** Static Python-to-C compiler using mypy for type analysis; offers good PEP-484 support, but with some reduced compatibility.
*   **Nuitka:** Static Python-to-C compiler; provides high language compliance.

## Support the Project

Support the Cython project via [GitHub Sponsors](https://github.com/users/scoder/sponsorship) or [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython).