# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write blazing-fast C extensions for Python as easily as you write Python itself.**  

[View the Cython Repository on GitHub](https://github.com/cython/cython)

Cython is a powerful, yet easy-to-use, language designed to bridge the gap between Python and C/C++. It's based on Pyrex but offers significant advancements in functionality and optimization. Cython translates your Python code (and code with C type declarations) into highly efficient C/C++ code.

**Key Features:**

*   **Performance:** Generate highly optimized C code.
*   **Ease of Use:** Write C extensions with Python-like syntax.
*   **C/C++ Integration:** Seamlessly call C functions and declare C types.
*   **Library Wrapping:** Ideal for wrapping external C libraries.
*   **CPython Compatibility:** Full runtime compatibility with all current and future versions of CPython.
*   **Platform Adaptation:** C compile-time adaptation to the target platform and Python version.
*   **Extensive User Base:** Thousands of libraries, packages and tools using Cython.

Cython is downloaded more than 60 million times per month on PyPI.

**Installation:**

If you have a C compiler, simply run:

```bash
pip install Cython
```

Otherwise, refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for instructions.

**Support the Project:**

*   [GitHub Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

**License:**

Cython is licensed under the permissive **Apache License**.

See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for details.

**Contributing:**

Want to contribute? Get started with the [contribution guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

**Comparisons to Other Python Compilers:**

Cython stands out among Python compilers by offering:

*   Fast, efficient, and compliant support for Python features.
*   "Generate once, compile everywhere" C code generation for reproducible results.
*   Seamless integration with C/C++ code.
*   Broad support for manual optimization.
*   Almost two decades of bug fixing and static code optimisations.

**Alternatives:**

*   [PyPy](https://www.pypy.org/)
*   [Numba](http://numba.pydata.org/)
*   [Pythran](https://pythran.readthedocs.io/)
*   [mypyc](https://mypyc.readthedocs.io/)
*   [Nuitka](https://nuitka.net/)

**Get the Full Source History:**

If you've downloaded a source archive and want the full Git history, ensure you have Git installed and run:

```bash
make repo
```

```