# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as you write Python itself, significantly boosting performance.**  Learn more about Cython and how it can accelerate your Python projects.  ([Original Repository](https://github.com/cython/cython))

## Key Features of Cython

*   **Python-to-C/C++ Compilation:** Translates Python code into highly efficient C/C++ code.
*   **C/C++ Integration:** Seamlessly calls C functions and declares C types for optimized code generation.
*   **Performance Enhancement:** Ideal for wrapping external C libraries and accelerating Python modules.
*   **Wide Adoption:**  Trusted by a large user base with thousands of libraries and packages.
*   **Compatibility:** Fully compatible with all versions of CPython, PyPy and Pyston.
*   **Mature and Stable:** Benefited from almost two decades of bug fixing and optimization.

## Installation

To install Cython, ensuring you have a C compiler installed, run:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.

See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for details.

## Contributing

We welcome contributions! Get started by reviewing the [CONTRIBUTING guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from Other Python Compilers

Cython stands out from other Python compilers by providing:

*   **Full Language Support:** Comprehensive support for almost all Python features.
*   **Runtime Compatibility:** Full runtime compatibility with all current and future CPython versions.
*   **Platform Adaptation:** C compile-time adaptation to the target platform and Python version.
*   **C/C++ Integration:** Seamless integration with C/C++ code.
*   **Fine-Grained Optimization:** Broad support for manual optimization and tuning down to the C level.

Similar projects include: PyPy, Numba, Pythran, mypyc, and Nuitka. Cython offers advantages in compatibility, optimization, and the ability to integrate with existing C/C++ code.

## Get the Full Source History

To retrieve the complete source history from a downloaded archive, use Git:

```bash
make repo
```