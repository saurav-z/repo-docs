# Cython: Write C Extensions for Python with Ease

Cython empowers you to write C extensions for Python as easily as you write Python itself, boosting performance and enabling seamless integration with C/C++ code.  [See the original repository here](https://github.com/cython/cython).

## Key Features

*   **Python to C/C++ Compilation:** Cython translates Python code into highly optimized C/C++ code, allowing you to create fast C modules.
*   **C Function & Type Integration:** Easily call C functions and declare C types within your Python code for fine-grained control and performance tuning.
*   **Seamless C/C++ Integration:**  Effortlessly wrap external C libraries and incorporate C/C++ code into your Python projects.
*   **Broad Python Language Support:**  Cython supports nearly all Python features, including dynamic features and introspection, for compatibility with various Python versions.
*   **Cross-Platform C Code Generation:** "Generate once, compile everywhere" C code generation ensuring reproducible performance results and easy testing.
*   **Extensive User Community:** Benefit from a large community with thousands of libraries, packages, and tools.
*   **Over Two Decades of Refinement:**  Benefit from years of bug fixing and static code optimizations.

## Installation

If you have a C compiler, install Cython with:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.

See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for details.

## Contributing

Interested in contributing to the Cython project?  Find helpful resources to get started here: [CONTRIBUTING.rst](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst)

## Differences from Other Python Compilers

Cython has outlived many other attempts at producing static compilers for Python. Cython offers several advantages over similar projects like:

*   **PyPy:** A Python implementation with a JIT compiler.
*   **Numba:** A Python extension with a JIT compiler for numerical code.
*   **Pythran:** A Python-to-C++ extension compiler for numerical computation.
*   **mypyc:** A static Python-to-C extension compiler based on mypy.
*   **Nuitka:** A static Python-to-C extension compiler.