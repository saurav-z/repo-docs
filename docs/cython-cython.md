# Cython: Write C Extensions for Python with Ease

**Supercharge your Python code and bridge the gap between Python and C/C++ with Cython, an optimizing compiler that brings the power of C to your Python projects.** [Learn more at the Cython GitHub Repository](https://github.com/cython/cython)

## Key Features

*   **Performance Boost:** Cython translates Python code into highly efficient C/C++ code, significantly speeding up execution.
*   **C/C++ Integration:** Seamlessly call C functions and declare C types within your Python code for fine-grained control.
*   **Ideal for Wrapping Libraries:** Easily wrap external C libraries for use in Python.
*   **Optimized C Modules:** Create fast C modules to accelerate your Python applications.
*   **Broad Compatibility:** Works with all CPython versions, ensuring stability and future-proofing your code.
*   **Mature and Stable:** Benefit from two decades of bug fixes and continuous optimization.

## Installation

If you have a C compiler, install Cython using pip:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.

See `LICENSE.txt <https://github.com/cython/cython/blob/master/LICENSE.txt>`_ for details.

## Contributing

Interested in contributing to the Cython project? Get started with this [guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from Other Python Compilers

Cython stands out from other Python compilers with its comprehensive support for Python features and its ability to optimize code at the C level.  Here's a comparison with similar projects:

*   **PyPy:** JIT compilation with runtime optimizations. Cons: non-CPython runtime, limited compatibility with CPython extensions.
*   **Numba:** JIT compiler for a subset of the language, focusing on numerical code. Cons: limited language support, large runtime dependency (LLVM).
*   **Pythran:** Static Python-to-C++ compiler, focused on numerical computation.
*   **mypyc:** Static Python-to-C extension compiler. Cons: no support for low-level optimizations.
*   **Nuitka:** Static Python-to-C extension compiler. Cons: no support for low-level optimizations and typing

Compared to these, Cython offers:

*   **Comprehensive Language Support:**  Fast, efficient support for almost all Python features.
*   **CPython Compatibility:** Full runtime compatibility with CPython, ensuring long-term stability.
*   **Reproducible Performance:** "Generate once, compile everywhere" C code generation for consistent results.
*   **C-API Support:** Includes support for PyPy and Pyston
*   **Manual Optimization:** Broad support for low-level tuning.
*   **Large Ecosystem:** A vast user base with thousands of libraries and tools.

## History

Cython originated from Pyrex, a language for writing Python extension modules developed by Greg Ewing. Pyrex was originally licensed "free of restrictions." Cython evolved from this foundation and continues to build upon its strong base.

You can find the latest version of Pyrex [here](https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/).

## Support Cython

*   **GitHub Sponsors:** [Support the Cython project via GitHub Sponsors](https://github.com/users/scoder/sponsorship)
*   **Tidelift:** [Support Cython via Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

## Resources

*   **Official Website:** [https://cython.org/](https://cython.org/)
*   **Documentation:** [https://docs.cython.org/](https://docs.cython.org/)
*   **GitHub Repository:** [https://github.com/cython/cython](https://github.com/cython/cython)
*   **Wiki:** [https://github.com/cython/cython/wiki](https://github.com/cython/cython/wiki)