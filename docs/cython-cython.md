# Cython: Write C Extensions for Python with Ease

**Cython empowers you to write C extensions for Python as easily as you write Python itself, enabling significant performance gains.**  [Learn more about Cython on GitHub](https://github.com/cython/cython).

## Key Features of Cython

*   **Optimizing Python Compiler:** Cython translates Python code into highly efficient C/C++ code.
*   **C/C++ Integration:** Seamlessly call C functions and declare C types for fine-grained control.
*   **Performance Enhancement:** Significantly speed up Python code execution and wrap external C libraries.
*   **Broad Compatibility:** Full runtime compatibility with CPython and support for other C-API implementations.
*   **Mature & Widely Used:**  Backed by a large user base and over two decades of development, with over 70 million monthly downloads on PyPI.

## Installation

To install Cython, simply use `pip`:

```bash
pip install Cython
```

For more detailed installation instructions, see the [Cython Installation Guide](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**.  For more details, see the [LICENSE.txt](https://github.com/cython/cython/blob/master/LICENSE.txt) file.

## Contributing

We welcome contributions! Learn how you can get involved in the [Cython project](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython stands out among other Python compilers, offering a unique blend of features and capabilities.  Here's a comparison:

| Feature                 | Cython                                                                                                                                                                                                                       | PyPy                                                                                                | Numba                                                                                                                      | Pythran                                                                                                                                                                                             | mypyc                                                                                                               | Nuitka                                                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Compilation Type        | Static                                                                                                                                                                                                                     | JIT                                                                                                 | JIT                                                                                                                        | Static                                                                                                                                                                                            | Static                                                                                                                 | Static                                                                                                                      |
| Language Compliance     | High, supports almost all Python features.                                                                                                                                                                                | Fully language compliant.                                                                             | Limited                                                                                                                    | Limited                                                                                                                                                                                           | Good support for language and PEP-484 typing.                                                                         | Highly language compliant.                                                                                               |
| C/C++ Integration       | Seamless                                                                                                                                                                                                                   | Good                                                                                                | Good                                                                                                                       | Seamless                                                                                                                                                                                             | No support for low-level optimizations and typing.                                                                      | No support for low-level optimizations and typing.                                                                       |
| Performance Optimization| Broad support for manual optimization and tuning down to the C level.                                                                                                                                                     | Runtime optimizations.                                                                                | Runtime optimizations.                                                                                                       | Targeted at numerical computation and can be used as a backend for NumPy code in Cython.                                                                                                       | Reasonable performance gains.                                                                                          | Reasonable performance gains.                                                                                               |
| Runtime Compatibility   | Full runtime compatibility with CPython.                                                                                                                                                                                  | Non-CPython runtime.                                                                               | Relatively large runtime dependency (LLVM).                                                                                |                                                                                                                                                                                                       | Reduced Python compatibility and introspection after compilation.                                                        |                                                                                                                            |
| Key Use Cases          | Wrapping external C libraries, creating fast C modules, optimizing existing Python code.                                                                                                                                   | Dynamic runtime optimizations.                                                                        | Mostly targets numerical code using NumPy.                                                                                    | Numerical computation and use as backend for NumPy code in Cython.                                                                                                                                |                                                                                                                       | Supports static application linking.                                                                                         |

## Support the Project

Show your support for Cython by becoming a sponsor via [Github Sponsors](https://github.com/users/scoder/sponsorship) or [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython).

## Further Resources

*   **Official Website:** [https://cython.org/](https://cython.org/)
*   **Documentation:** [https://docs.cython.org/](https://docs.cython.org/)
*   **GitHub Repository:** [https://github.com/cython/cython](https://github.com/cython/cython)
*   **Wiki:** [https://github.com/cython/cython/wiki](https://github.com/cython/cython/wiki)