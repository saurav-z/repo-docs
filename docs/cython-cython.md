# Cython: Write C Extensions for Python with Ease

**Supercharge your Python code with Cython, the optimizing compiler that bridges the gap between Python and C/C++ for blazing-fast performance.**  Find the original repository at [https://github.com/cython/cython](https://github.com/cython/cython).

## Key Features of Cython:

*   **Python to C/C++ Translation:** Cython translates Python code into efficient C/C++ code, unlocking significant performance gains.
*   **C/C++ Integration:** Seamlessly call C functions and declare C types within your Python code for fine-grained control.
*   **Manual Tuning:** Fine-tune code with Cython to generate highly optimized C code.
*   **Ideal for:** Wrapping external C libraries and creating fast C modules to accelerate Python applications.
*   **Mature and Widely Used:** Cython boasts a massive user base, over 70 million monthly downloads on PyPI, and two decades of development.
*   **Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython, along with support for other C-API implementations, including PyPy and Pyston.
*   **Reproducible Performance:** "Generate once, compile everywhere" C code generation that allows for reproducible performance results and testing.

## Installation

If you have a C compiler, install Cython using:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive [Apache License](https://github.com/cython/cython/blob/master/LICENSE.txt).

## Contributing

Interested in contributing to the Cython project? Get started with this [guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython stands out from other Python compilers by providing:

*   Fast, efficient, and highly compliant support for almost all Python language features, including dynamic features and introspection.
*   Broad support for manual optimization and tuning down to the C level.
*   C compile-time adaptation to the target platform and Python version.
*   A large user base with thousands of libraries, packages, and tools.

### Comparison with Similar Projects

*   **PyPy:** JIT compiler with runtime optimizations.  Pros: Good integration with C/C++ code. Cons: Non-CPython runtime, limited compatibility with CPython extensions.
*   **Numba:** JIT compiler for a subset of Python, primarily for numerical code using NumPy.  Pros: JIT compilation with runtime optimizations. Cons: Limited language support.
*   **Pythran:** Static Python-to-C++ extension compiler, mainly for numerical computation.
*   **mypyc:** Static Python-to-C extension compiler. Pros: Good support for PEP-484 typing, good type inference. Cons: No support for low-level optimizations.
*   **Nuitka:** Static Python-to-C extension compiler.  Pros: Highly language compliant, reasonable performance gains. Cons: No support for low-level optimizations.

## Support the Cython Project

You can support the Cython project through:

*   [Github Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

## Legacy: Based on Pyrex

Cython was originally based on `Pyrex <https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/>`_ by Greg Ewing.