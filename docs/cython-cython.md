# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as you write Python itself, boosting performance and enabling seamless integration with C/C++ code.**

[View the original repository on GitHub](https://github.com/cython/cython)

## Key Features

*   **Python-to-C/C++ Compilation:** Translates Python code into highly efficient C/C++ code for significant speedups.
*   **C/C++ Integration:**  Easily call C functions and declare C types within your Python code.
*   **Fine-Grained Tuning:** Offers broad to fine-grained manual tuning for optimal C code generation.
*   **Ideal for Wrapping C Libraries:**  Simplifies the process of integrating external C libraries into your Python projects.
*   **Optimized Modules:** Create fast C modules to accelerate the execution of Python code.
*   **CPython Compatibility:** Full runtime compatibility with all CPython versions.
*   **Cross-Platform Support:**  "Generate once, compile everywhere" C code generation for reproducible results.
*   **Large User Base:**  Leveraged by thousands of libraries, packages, and tools.
*   **Mature and Stable:**  Benefit from over two decades of bug fixes and static code optimizations.

## Installation

If you have a C compiler, installing Cython is straightforward:

```bash
pip install Cython
```

For detailed installation instructions, please see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## Contributing

We welcome contributions!  Get started by reviewing the [contributing guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## License

Cython is licensed under the permissive **Apache License**.

## Resources

*   **Official Website:** [https://cython.org/](https://cython.org/)
*   **Documentation:** [https://docs.cython.org/](https://docs.cython.org/)
*   **GitHub Repository:** [https://github.com/cython/cython](https://github.com/cython/cython)
*   **Wiki:** [https://github.com/cython/cython/wiki](https://github.com/cython/cython/wiki)
*   **PyPI Stats:** [https://pypistats.org/packages/cython](https://pypistats.org/packages/cython)
*   **Support Cython:**
    *   [GitHub Sponsors](https://github.com/users/scoder/sponsorship)
    *   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

## Comparison with Other Python Compilers/Implementations

Cython has been a leading project in the Python compilation space, outliving most other attempts. Here's a brief comparison:

*   **PyPy:** A Python implementation with a JIT compiler.
*   **Numba:** A Python extension with a JIT compiler for a subset of the language, based on the LLVM compiler infrastructure.
*   **Pythran:** A static Python-to-C++ extension compiler for numerical computation.
*   **mypyc:** A static Python-to-C extension compiler, based on the mypy static Python analyser.
*   **Nuitka:** A static Python-to-C extension compiler.

Cython distinguishes itself with its:

*   Fast, compliant support for almost all Python features
*   CPython compatibility
*   C compile time adaptation to the target platform and Python version
*   Seamless integration with C/C++ code
*   Broad support for manual optimisation and tuning down to the C level