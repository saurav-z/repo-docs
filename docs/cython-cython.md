# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as writing Python itself, significantly boosting your code's performance.**

[![PyPI downloads](https://img.shields.io/pypi/dm/Cython.svg?style=flat-square)](https://pypistats.org/packages/cython)
[![GitHub stars](https://img.shields.io/github/stars/cython/cython?style=flat-square)](https://github.com/cython/cython)
[![GitHub forks](https://img.shields.io/github/forks/cython/cython?style=flat-square)](https://github.com/cython/cython)

[**View the Cython Repository on GitHub**](https://github.com/cython/cython)

Cython is a powerful language that bridges the gap between Python and C/C++, allowing you to create high-performance extensions for your Python code. It's based on Pyrex but offers advanced features and optimizations. With over 60 million downloads per month on PyPI, Cython is a trusted tool for accelerating Python applications.

## Key Features

*   **Python-to-C/C++ Compilation:** Translates Python code into efficient C/C++ code for significant speed improvements.
*   **C/C++ Integration:** Seamlessly calls C functions and declares C types, enabling you to wrap existing C libraries and create fast C modules.
*   **Performance Optimization:** Provides tools for manual optimization and tuning at the C level.
*   **Broad Python Compatibility:** Offers excellent support for almost all Python language features, including dynamic features and introspection.
*   **Cross-Platform Compatibility:** "Generate once, compile everywhere" C code generation for reproducible performance results and testing.
*   **Mature and Widely Used:** A large user base with thousands of libraries, packages, and tools, refined through almost two decades of bug fixing and static code optimizations.

## Installation

If you have a C compiler already, install Cython with:

```bash
pip install Cython
```

For more detailed installation instructions, please refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## Support the Project

Support the Cython project by:

*   [GitHub Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

## License

Cython is licensed under the permissive **Apache License**.

See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for details.

## Contributing

Interested in contributing to the Cython project? Get started with the [contributing guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Comparisons with other Python Compilers

Cython stands out from other Python compilers by offering a unique blend of performance, compatibility, and ease of use.  Here's how Cython compares to other projects:

*   [PyPy](https://www.pypy.org/): A Python implementation with a JIT compiler, offering runtime optimizations and good C/C++ integration but with potential compatibility limitations.
*   [Numba](http://numba.pydata.org/): A Python extension with a JIT compiler based on LLVM, primarily targeting numerical code with runtime optimizations but with limited language support.
*   [Pythran](https://pythran.readthedocs.io/): A static Python-to-C++ extension compiler focused on numerical computation, best used as a NumPy backend in Cython.
*   [mypyc](https://mypyc.readthedocs.io/): A static Python-to-C extension compiler based on mypy, supporting PEP-484 typing with good type inference, but with less Python compatibility.
*   [Nuitka](https://nuitka.net/): A static Python-to-C extension compiler that offers high language compliance and reasonable performance gains.