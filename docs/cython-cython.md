# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as you write Python itself, unlocking significant performance gains.**  Visit the original repository on [GitHub](https://github.com/cython/cython).

## Key Features

*   **Python to C/C++ Compilation:** Translates Python code into C/C++ code for optimized performance.
*   **C and C++ Integration:** Seamlessly calls C functions and declares C types within your Python code.
*   **Fine-Grained Tuning:**  Allows for manual optimization, enabling the compiler to generate highly efficient C code.
*   **Ideal for Wrapping C Libraries:** Simplifies the process of integrating existing C libraries into Python projects.
*   **Fast C Modules:**  Accelerates the execution of Python code through efficient C modules.
*   **Broad Python Language Support:**  Supports almost all Python language features, including dynamic features and introspection.
*   **CPython Compatibility:** Full runtime compatibility with all current and future versions of CPython.
*   **"Generate Once, Compile Everywhere":** C code generation allows for reproducible performance results and testing
*   **Platform and Python Version Adaptation:**  C compile time adaptation to the target platform and Python version
*   **C-API Implementation Support:** Support for other C-API implementations, including PyPy and Pyston
*   **Extensive Optimization Support:**  Broad support for manual optimization and tuning down to the C level.
*   **Large User Base:**  Used by thousands of libraries, packages, and tools.
*   **Mature & Stable:**  Over two decades of bug fixing and static code optimizations.

## Installation

If you have a C compiler installed, install Cython with:

```bash
pip install Cython
```

Otherwise, refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for detailed instructions.

## License

Cython is licensed under the permissive **Apache License**.  

See the [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) file for more details.

## Contributing

Interested in contributing to the Cython project?  Find help to get you started [here](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Downloads & Support

Cython has more than 70 million downloads per month on PyPI.  You can support the Cython project via:

*   [Github Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

## Differences Compared to Other Python Compilers

Cython, developed since the early 2000s, offers key advantages over other Python compilers, including:

*   **PyPy:** A Python implementation with a JIT compiler (runtime optimizations).
*   **Numba:**  A Python extension with a JIT compiler for a subset of the language, focused on numerical code.
*   **Pythran:** A static Python-to-C++ extension compiler focused on numerical computation (often used as a backend for NumPy code in Cython).
*   **mypyc:** A static Python-to-C extension compiler, based on mypy, using PEP-484 type annotations for optimizations.
*   **Nuitka:**  A static Python-to-C extension compiler, known for its language compliance and the ability to bundle library dependencies into a self-contained executable.

Cython's distinct advantages include its broad Python language support, integration capabilities, and extensive optimization options.

## Resources

*   **Official Website:** [https://cython.org/](https://cython.org/)
*   **Documentation:** [https://docs.cython.org/](https://docs.cython.org/)
*   **GitHub Repository:** [https://github.com/cython/cython](https://github.com/cython/cython)
*   **Wiki:** [https://github.com/cython/cython/wiki](https://github.com/cython/cython/wiki)