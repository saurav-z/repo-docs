# Cython: Write C Extensions for Python with Ease 🚀

**Cython empowers you to write C extensions for Python as effortlessly as you write Python itself, accelerating your code and bridging the gap between high-level Python and low-level C.** [View the original repository](https://github.com/cython/cython).

## Key Features

*   **Python-to-C/C++ Compilation:** Translates Python code into highly optimized C/C++ code.
*   **C Function & Type Integration:** Seamlessly calls C functions and declares C types for fine-grained control.
*   **Performance Optimization:**  Allows for manual tuning to generate efficient C code.
*   **Ideal for Wrapping C Libraries:**  Perfect for integrating external C libraries.
*   **Fast C Modules:** Accelerates the execution of Python code.
*   **CPython Compatibility:** Full runtime compatibility with all CPython versions.
*   **Mature & Widely Used:**  Boasts over 70 million monthly downloads on PyPI.

## Installation

If you have a C compiler, simply run:

```bash
pip install Cython
```

Otherwise, refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**. See [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) for details.

## Contributing

Interested in contributing?  Find helpful resources to get started at [CONTRIBUTING.rst](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython stands out amongst other Python compilers, offering a robust, flexible, and efficient solution:

*   **PyPy:** A Python implementation with a JIT compiler.
    *   **Pros:** JIT compilation, full language compliance, good C/C++ integration.
    *   **Cons:** Non-CPython runtime, resource usage, limited extension compatibility.
*   **Numba:** JIT compiler for a subset of Python, optimized for numerical code.
    *   **Pros:** JIT compilation with runtime optimisations.
    *   **Cons:** Limited language support, LLVM dependency.
*   **Pythran:** Static Python-to-C++ compiler, focused on numerical computation.
*   **mypyc:** Static Python-to-C extension compiler based on mypy.
    *   **Pros:** Good PEP-484 typing support, good type inference.
    *   **Cons:** No low-level optimisations, reduced Python compatibility.
*   **Nuitka:** Static Python-to-C extension compiler.
    *   **Pros:** Highly language compliant, supports static application linking.
    *   **Cons:** No low-level optimisations.

**Why Choose Cython?**

*   **Comprehensive Language Support:**  Handles almost all Python features, including dynamic aspects.
*   **CPython Compatibility:** Works seamlessly with all CPython versions.
*   **Reproducible Performance:** Generates C code for consistent results.
*   **Platform Adaptation:** Adapts at compile time to the target platform and Python version.
*   **C-API Implementation Support:** Includes support for PyPy and Pyston.
*   **C/C++ Integration:**  Easily integrates with existing C/C++ code.
*   **Fine-Grained Control:** Offers broad support for manual optimization.
*   **Large Ecosystem:**  Supported by thousands of libraries, packages, and tools.
*   **Maturity:** Benefit from over two decades of bug fixes and optimizations.

## History - Based on Pyrex

Cython builds upon the foundation laid by Pyrex, a language for writing Python extension modules developed by Greg Ewing. Pyrex was licensed "free of restrictions" allowing for usage, redistribution, modification, and distribution of modified versions.