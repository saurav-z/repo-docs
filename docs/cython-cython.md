# Cython: Write C Extensions for Python with Ease

**Cython is a powerful optimizing compiler that transforms Python code into efficient C/C++ code, making it easy to create high-performance extensions and wrap external C libraries.**  Learn how to leverage the speed of C with the convenience of Python using Cython.  [Learn more on GitHub!](https://github.com/cython/cython)

## Key Features

*   **Python-to-C/C++ Translation:** Converts Python code to C/C++ code, enabling significant speed improvements.
*   **C Function and Type Integration:** Seamlessly call C functions and declare C types within your Python code for fine-grained control.
*   **Optimized C Code Generation:** Allows for manual tuning, generating highly efficient C code for optimal performance.
*   **Ideal for C Library Wrapping:** Simplifies the process of integrating and using existing C libraries.
*   **High-Performance Modules:** Create fast C modules to accelerate Python code execution.
*   **Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython.
*   **Manual Optimization:** Offers broad support for manual optimization and tuning down to the C level.

## Installation

If you have a C compiler, installation is straightforward:

```bash
pip install Cython
```

Otherwise, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html) for detailed instructions.

## License

Cython is licensed under the permissive **Apache License**.

See `LICENSE.txt <https://github.com/cython/cython/blob/master/LICENSE.txt>`_.

## Contributing

Interested in contributing to Cython?  Get started with [this guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython has outlived most other attempts at producing static compilers for the Python language.  Here is a comparison:

| Feature          | Cython                               | Other Compilers (e.g., PyPy, Numba, mypyc, Nuitka)                     |
|-------------------|---------------------------------------|------------------------------------------------------------------------|
| Language Support | High, almost all Python features      | Varies, often limited                                                 |
| CPython          | Full compatibility with CPython         | Can be non-CPython runtime, compatibility varies                     |
| Optimization     | Fine-grained manual optimization       | Often runtime optimizations, limited low-level control                |

## Support the Project

You can support the Cython project via [GitHub Sponsors](https://github.com/users/scoder/sponsorship) or [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython).

## Pyrex (The Original Project)

Cython was originally based on `Pyrex <https://www.cosc.canterbury.ac.nz/~greg/python/Pyrex/>`_ by Greg Ewing.