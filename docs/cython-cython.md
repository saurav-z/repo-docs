# Cython: Write Pythonic Code that Runs at C Speed

**Cython is a powerful optimizing static compiler that transforms Python code into highly efficient C/C++ code, bridging the gap between Python's ease of use and C's performance.**  [Learn more on the original Cython repo](https://github.com/cython/cython).

## Key Features:

*   **Performance Boost:** Significantly speeds up Python code execution through compilation to C/C++.
*   **Pythonic Syntax:** Write C extensions as easily as writing Python itself.
*   **C/C++ Integration:** Seamlessly call C functions and declare C types for fine-grained optimization.
*   **External Library Wrapping:** Easily wrap and utilize external C libraries within your Python code.
*   **Optimized C Code Generation:**  Generate highly efficient C code through manual tuning options.
*   **Wide Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython and support for other C-API implementations, including PyPy and Pyston.
*   **Mature and Widely Used:**  Benefiting from over two decades of development, Cython boasts a large user base and is integrated in thousands of libraries, packages, and tools.

## Installation

If you have a C compiler already, install Cython easily with:

```bash
pip install Cython
```

Otherwise, consult the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for detailed instructions.

## License

Cython is licensed under the permissive **Apache License**.

See the full license in [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt).

## Contributing

Want to contribute?  Find helpful information to get started in the [CONTRIBUTING guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython stands out from other Python compilers. Here is a comparison.

### Similar Projects

*   **PyPy:** JIT compiler with runtime optimizations (non-CPython runtime, resource-intensive, limited extension compatibility).
*   **Numba:** JIT compiler for a subset of Python, targets numerical code using LLVM (limited language support, dependency on LLVM, performance can be unpredictable).
*   **Pythran:** Static Python-to-C++ compiler for numerical computation (best used as a NumPy backend in Cython).
*   **mypyc:** Static Python-to-C compiler based on mypy (good PEP-484 support, lacks low-level optimization, reduced Python compatibility).
*   **Nuitka:** Static Python-to-C compiler (high language compliance, lacks low-level optimization and typing).

### Advantages of Cython

*   **Full Python Feature Support:** Supports nearly all Python features.
*   **CPython Compatibility:**  Maintains runtime compatibility with all CPython versions.
*   **Reproducible Performance:** "Generate once, compile everywhere" C code generation.
*   **Platform Adaptation:** C compile-time adaptation to target platform and Python version.
*   **C/C++ Integration:** Seamless integration with existing C/C++ code.
*   **Fine-Grained Optimization:** Supports manual optimization and tuning down to the C level.
*   **Mature and Stable:**  Benefiting from over two decades of development with a large user base.

## About Pyrex (the original basis of Cython)

Cython evolved from Pyrex, a project by Greg Ewing.

This is a development version of Pyrex, a language for writing Python extension modules.

*   Doc/About.html for a description of the language
*   INSTALL.txt for installation instructions
*   USAGE.txt for usage instructions
*   Demos for usage examples

Copyright stuff: Pyrex is free of restrictions. You may use, redistribute, modify and distribute modified versions.

The latest version of Pyrex can be found [here](https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/).

## Additional Resources

*   **Official Website:** [https://cython.org/](https://cython.org/)
*   **Documentation:** [https://docs.cython.org/](https://docs.cython.org/)
*   **GitHub Repository:** [https://github.com/cython/cython](https://github.com/cython/cython)
*   **Wiki:** [https://github.com/cython/cython/wiki](https://github.com/cython/cython/wiki)

## Support the Cython Project

*   **Github Sponsors:** [https://github.com/users/scoder/sponsorship](https://github.com/users/scoder/sponsorship)
*   **Tidelift:** [https://tidelift.com/subscription/pkg/pypi-cython](https://tidelift.com/subscription/pkg/pypi-cython)