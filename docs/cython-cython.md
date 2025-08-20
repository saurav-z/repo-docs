# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write blazing-fast C extensions for Python with a syntax that feels just like Python!** ([See the original repository](https://github.com/cython/cython))

## Key Features

*   **Performance Boost:** Cython translates Python code to highly optimized C/C++ code, dramatically speeding up execution, making it ideal for performance-critical sections of your Python code.
*   **Seamless C/C++ Integration:** Easily call C functions and declare C types within your Python code, enabling you to wrap existing C libraries and leverage their power.
*   **Fine-Grained Control:** Cython allows you to fine-tune your code with manual optimizations, giving you control over the generated C code for maximum efficiency.
*   **CPython Compatibility:** Full runtime compatibility with CPython, ensuring your Cython code works with all current and future versions of the standard Python implementation.
*   **Large Ecosystem:** Benefit from a massive user base and integrate with thousands of existing libraries, packages, and tools.
*   **Mature and Stable:** Built upon the foundation of Pyrex and refined over two decades, Cython offers a reliable and well-tested solution.

## Installation

If you have a C compiler, installation is straightforward:

```bash
pip install Cython
```

For more detailed installation instructions, please see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive [Apache License](https://github.com/cython/cython/blob/master/LICENSE.txt).

## Contributing

Want to contribute to the Cython project?  Get started with some [helpful resources](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython offers advantages over other Python compilers:

*   **PyPy:** While PyPy offers JIT compilation, it is a non-CPython runtime with compatibility limitations.
*   **Numba:** Numba focuses on numerical code and has runtime dependencies.
*   **Pythran:** Primarily aimed at numerical computation, often used as a backend for NumPy code in Cython.
*   **mypyc:** Relies on type annotations, offering good type inference but with reduced Python compatibility.
*   **Nuitka:** Supports static application linking but lacks low-level optimization features.

Cython excels due to its extensive Python language support, tight CPython integration, and flexible optimization capabilities.

## Get Involved

*   **Official Website:** [https://cython.org/](https://cython.org/)
*   **Documentation:** [https://docs.cython.org/](https://docs.cython.org/)
*   **GitHub Repository:** [https://github.com/cython/cython](https://github.com/cython/cython)
*   **Wiki:** [https://github.com/cython/cython/wiki](https://github.com/cython/cython/wiki)
*   **Support the Project:** You can support the Cython project via [Github Sponsors](https://github.com/users/scoder/sponsorship) or [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython).