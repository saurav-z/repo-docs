# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write fast C extensions for Python as easily as you write Python itself!**  [Learn more at the Cython GitHub repository](https://github.com/cython/cython).

## Key Features

*   **High-Performance Python Extensions:** Translate Python code into highly efficient C/C++ code for significant performance gains.
*   **Seamless C/C++ Integration:** Easily call C functions and declare C types within your Python code.
*   **Fine-Grained Control:**  Offers broad and fine-grained manual tuning for optimized C code generation.
*   **Ideal for Wrapping C Libraries:**  Perfect for creating Python bindings for existing C libraries.
*   **Cross-Platform Compatibility:** Generate C code that adapts to the target platform and Python version.
*   **Extensive Compatibility:** Full runtime compatibility with all CPython versions.
*   **Large Community & Ecosystem:** Benefiting from a large user base with thousands of libraries, packages, and tools.

## What is Cython?

Cython is an optimizing static compiler for both the Python programming language and the extended Cython programming language (based on Pyrex). It translates Python code into C code, allowing you to create high-performance Python extensions and integrate seamlessly with existing C/C++ libraries. Cython supports a wide array of Python features, along with options for fine-grained control and low-level optimizations to achieve significant speed improvements.

## Installation

If you have a C compiler installed, you can install Cython using pip:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.

See the full license details in [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt).

## Contributing

We welcome contributions!  Get started by reviewing the [contribution guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from Other Python Compilers

Cython stands out from other Python compilers due to:

*   **Mature Development:** Cython has been actively developed for over two decades.
*   **Dynamic Language Features:** Supports a wide array of dynamic Python language features.
*   **Reproducible Results:** "Generate once, compile everywhere" C code generation.

Compared to projects like PyPy, Numba, Pythran, mypyc, and Nuitka, Cython offers a unique combination of features, including:

*   **High language compliance**
*   **Runtime compatibility with CPython**
*   **Support for C/C++ code**
*   **Manual optimization options**

## Credits

Cython was originally based on Pyrex, a project by Greg Ewing.