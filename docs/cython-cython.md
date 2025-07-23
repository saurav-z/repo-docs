# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write blazing-fast C extensions for Python as easily as you write Python itself.**  [Check out the original repository](https://github.com/cython/cython).

## Key Features

*   **Python to C/C++ Compilation:** Transpiles Python code into highly efficient C/C++ code.
*   **C/C++ Integration:** Seamlessly calls C functions and declares C types for superior performance.
*   **Ideal for Performance Enhancement:**  Perfect for wrapping external C libraries and accelerating Python code execution.
*   **Wide Adoption:**  Cython boasts millions of downloads per month.

## Installation

If you have a C compiler installed, simply run:

```bash
pip install Cython
```

Otherwise, consult the [installation guide](https://docs.cython.org/en/latest/src/quickstart/install.html) for detailed instructions.

## Licensing

Cython is licensed under the permissive **Apache License**. See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for details.

## Contributing

Interested in contributing to Cython? Get started with the [contribution guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython stands out among Python compilers, offering a unique blend of features and advantages:

*   **PyPy:** JIT compilation, fully language compliant, good C/C++ integration (Cons: non-CPython runtime, resource usage, limited CPython extension compatibility).
*   **Numba:** JIT compiler for numerical code using LLVM (Cons: limited language support, runtime dependency, performance variability).
*   **Pythran:** Static Python-to-C++ compiler for numerical computation. (Best used as a NumPy backend within Cython).
*   **mypyc:** Static Python-to-C extension compiler (Cons: no low-level optimizations, opinionated type interpretation, reduced Python compatibility).
*   **Nuitka:** Static Python-to-C extension compiler (Cons: no low-level optimizations, reduced Python compatibility).

**Cython's distinct advantages include:**

*   Excellent support for almost all Python language features.
*   Full runtime compatibility with all still-in-use and future versions of CPython.
*   C code generation for reproducible performance.
*   Seamless integration with C/C++ code.
*   Extensive support for manual optimization.
*   Mature codebase with a large user base and robust libraries.

## Support Cython

Support the Cython project via [Github Sponsors](https://github.com/users/scoder/sponsorship) or [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython).

## Get the Full Source History

To retrieve the full source history, ensure Git is installed, navigate to the Cython source directory and run:

```bash
make repo
```

---