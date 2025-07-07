# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as you write Python itself, offering performance and seamless C/C++ integration.** Explore the full power of Cython and optimize your Python code. ([Original Repository](https://github.com/cython/cython))

## Key Features

*   **Python-to-C/C++ Compilation:** Transpiles Python code into highly efficient C/C++ code.
*   **C/C++ Integration:** Seamlessly calls C functions and declares C types, enabling powerful optimizations.
*   **Performance Boost:** Accelerates Python code execution, making it ideal for computationally intensive tasks.
*   **External Library Wrapping:** Simplifies the wrapping of external C libraries for use within Python.
*   **Wide Adoption:** Used by thousands of libraries, packages, and tools with over 60 million monthly downloads on PyPI.
*   **CPython Compatibility:** Full runtime compatibility with current and future versions of CPython.
*   **Static Code Optimizations:** Benefit from nearly two decades of bug fixing and static code optimization.

## Installation

If you have a C compiler, install Cython with a simple command:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## Licensing

Cython is licensed under the permissive **Apache License**.

## Contributing

Want to contribute to the Cython project? Get started with [help to get you started](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Comparison with other Python Compilers

Cython stands out from other Python compilers. Here's a comparison:

*   [PyPy](https://www.pypy.org/): JIT compiler, good C/C++ integration, but non-CPython runtime.
*   [Numba](http://numba.pydata.org/): JIT compiler for a subset of Python, focused on numerical code.
*   [Pythran](https://pythran.readthedocs.io/): Static Python-to-C++ compiler, mainly for numerical computation.
*   [mypyc](https://mypyc.readthedocs.io/): Static Python-to-C compiler using PEP-484 type annotations.
*   [Nuitka](https://nuitka.net/): Static Python-to-C compiler, with good language compliance.

In comparison, Cython provides:

*   Fast, efficient, and highly compliant support for Python features.
*   Full runtime compatibility with all CPython versions.
*   "Generate once, compile everywhere" code generation for reproducible performance.
*   Seamless integration with C/C++ code.
*   Broad support for manual optimization.

## Get the full source history

To get the full source history from a downloaded source archive, install git, then step into the base directory of the Cython source distribution and type:

```bash
make repo
```