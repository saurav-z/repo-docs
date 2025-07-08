# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as you write Python itself, boosting performance and enabling seamless integration with C/C++ code.**  Explore the power of Cython to accelerate your Python projects!  ([See the original repository](https://github.com/cython/cython))

## Key Features

*   **Python-to-C/C++ Compilation:** Cython translates Python code into highly efficient C/C++ code.
*   **C Function and Type Declarations:**  Integrate C functions and declare C types directly within your code for optimized performance.
*   **External C Library Wrapping:**  Easily wrap and utilize external C libraries within your Python projects.
*   **Speed Up Python Modules:**  Create fast C modules to significantly improve the execution speed of your Python code.
*   **Broad Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython.
*   **"Generate Once, Compile Everywhere":** C code generation that allows for reproducible performance results and testing.
*   **Seamless Integration with C/C++:**  Easily integrate with existing C/C++ codebases.
*   **Mature and Widely Used:** Benefiting from almost two decades of bug fixing and static code optimizations with over 60 million monthly downloads on PyPI.

## Installation

If you have a C compiler, install Cython with:

```bash
pip install Cython
```

For more detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## Licensing

Cython is licensed under the permissive **Apache License**. See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for details.

## Contributing

Interested in contributing to the Cython project? Find out how to get started [here](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Supporting Cython

You can support the Cython project through:

*   [GitHub Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

## Differences from Other Python Compilers

Cython has established itself as a robust and well-supported solution for creating Python extensions.  Here's a comparison with other relevant projects:

*   **PyPy:** Python implementation with a JIT compiler.
*   **Numba:** JIT compiler for a subset of the language, focused on numerical code.
*   **Pythran:** Static Python-to-C++ extension compiler, primarily for numerical computation.
*   **mypyc:** Static Python-to-C extension compiler, based on mypy.
*   **Nuitka:** Static Python-to-C extension compiler.

## Get Full Source History
To get the full source history from a downloaded source archive, make sure you have git installed, then step into the base directory of the Cython source distribution and type::

```bash
make repo
```