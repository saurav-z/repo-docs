# Cython: Write C Extensions for Python with Ease

**Cython empowers Python developers to write high-performance C extensions as easily as writing Python itself.**  Enhance your Python code with seamless C integration and fine-grained control for optimal performance. Explore the official [Cython repository](https://github.com/cython/cython) to learn more!

## Key Features of Cython

*   **Python-like Syntax:** Write C extensions with a familiar Python syntax, making the transition smooth and intuitive.
*   **C/C++ Integration:** Seamlessly call C functions and declare C types directly within your Python code for optimized performance.
*   **Performance Optimization:** Fine-tune your code with manual optimizations, generating highly efficient C code.
*   **C Library Wrapping:** Easily wrap external C libraries for use in your Python projects.
*   **Fast C Modules:** Create high-speed C modules to accelerate the execution of your Python code.
*   **Cross-Platform Compilation:** Cython generates "generate once, compile everywhere" C code for reproducible performance results.
*   **CPython Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython.
*   **Mature & Widely Used:** Benefit from over two decades of development, a large user base, and extensive library support.

## Installation

If you have a C compiler installed, installation is as simple as running:

```bash
pip install Cython
```

For detailed installation instructions, see the [Cython installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## Usage & Resources

*   **Official Website:** [https://cython.org/](https://cython.org/)
*   **Documentation:** [https://docs.cython.org/](https://docs.cython.org/)
*   **GitHub Repository:** [https://github.com/cython/cython](https://github.com/cython/cython)
*   **Wiki:** [https://github.com/cython/cython/wiki](https://github.com/cython/cython/wiki)

## Support the Project

The Cython project is supported by the community.  You can contribute through:

*   **GitHub Sponsors:** [https://github.com/users/scoder/sponsorship](https://github.com/users/scoder/sponsorship)
*   **Tidelift:** [https://tidelift.com/subscription/pkg/pypi-cython](https://tidelift.com/subscription/pkg/pypi-cython)

## License

Cython is licensed under the permissive **Apache License**. See `LICENSE.txt <https://github.com/cython/cython/blob/master/LICENSE.txt>`_ for details.

## Contributing

Interested in contributing to Cython?  Find resources to get started here: [https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst)

## Comparison with other Python Compilers

Cython has outlived most other attempts at producing static compilers for Python.  Here's how it compares to some similar projects:

*(This section can be expanded with the information from the original README, focusing on the key pros and cons of each.  It is important to include keywords like "JIT compiler," "static compiler," "Python-to-C," and names of other compilers. Example:)*

*   **PyPy:** A Python implementation with a JIT compiler (Pros: JIT compilation, good C integration; Cons: non-CPython runtime).
*   **Numba:**  A JIT compiler for a subset of Python, targeting numerical code (Pros: JIT compilation; Cons: limited language support).
*   **Pythran:** A static Python-to-C++ extension compiler for numerical computation (Best used as a backend for NumPy code in Cython).
*   **mypyc:** A static Python-to-C compiler based on mypy (Pros: Good support for typing; Cons: No low-level optimizations, reduced Python compatibility).
*   **Nuitka:** A static Python-to-C extension compiler (Pros: highly language compliant; Cons: No low-level optimizations).

**Key Advantages of Cython:**  (Summarizing from the original README)  Fast, efficient, highly compliant with Python features, full CPython compatibility, "generate once, compile everywhere" C code generation, seamless integration with C/C++ code, broad support for manual optimization, and a large user base.

## History: Based on Pyrex

Cython was originally based on [Pyrex](https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/) by Greg Ewing.