# Cython: Write C Extensions for Python with Ease

**Cython is a powerful Python compiler that allows you to write C extensions for Python as easily as you write Python itself.** You can find the source code and more information on the project's GitHub repository: [https://github.com/cython/cython](https://github.com/cython/cython).

## Key Features

*   **Python to C/C++ Compilation:** Translates Python code into efficient C/C++ code, allowing for significant performance improvements.
*   **C/C++ Integration:** Seamlessly integrates with C/C++ code, enabling you to call C functions and declare C types within your Python code.
*   **Ideal for Extension Modules:** Perfect for wrapping external C libraries and creating fast C modules to accelerate Python code execution.
*   **Broad Compatibility:** Supports almost all Python language features and is fully compatible with all CPython versions.
*   **Optimization Capabilities:** Provides extensive support for manual optimization and tuning at the C level.

## Installation

If you have a C compiler installed, install Cython using:

```bash
pip install Cython
```

Otherwise, consult the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for detailed instructions.

## License

Cython is licensed under the permissive **Apache License**.

## Contributing

Interested in contributing to the Cython project?  Get started with the [contributing guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences to Other Python Compilers

Cython distinguishes itself from other Python compilers, such as:

*   **PyPy:** JIT compiler with runtime optimizations, but non-CPython runtime and limited compatibility with CPython extensions.
*   **Numba:** JIT compiler focused on numerical code, with limited language support and a runtime dependency on LLVM.
*   **Pythran:** Static Python-to-C++ compiler for numerical computation, often used as a backend for NumPy code in Cython.
*   **mypyc:** Static Python-to-C extension compiler using PEP-484 type annotations, with limited low-level optimization support.
*   **Nuitka:** Static Python-to-C extension compiler, with reasonable performance gains, but lacking low-level optimization and typing support.

Compared to these, Cython provides a comprehensive solution for efficient C extension development.