# Cython: Write C Extensions for Python with Ease

**Cython empowers Python developers to write C extensions as effortlessly as writing Python itself, boosting performance and enabling seamless integration with C/C++ code.** [See the original repository](https://github.com/cython/cython).

Cython is a powerful optimizing compiler that translates Python code into highly efficient C/C++ code. It allows you to combine the ease of Python with the speed of C, making it ideal for:

*   **Accelerating Python Code:** Cython can dramatically speed up your Python applications by compiling performance-critical sections to C.
*   **Wrapping C/C++ Libraries:** Easily integrate and utilize existing C and C++ libraries within your Python projects.
*   **Fine-Grained Control:** Cython allows for both high-level Python-like coding and low-level C tuning, giving you complete control over optimization.
*   **Compatibility:** Full runtime compatibility with all still-in-use and future versions of CPython

## Key Features

*   **Python to C/C++ Translation:**  Converts Python code into efficient C/C++ code.
*   **C Function & Type Support:**  Allows calling C functions and declaring C types for fine-grained optimization.
*   **Seamless C/C++ Integration:**  Easily wraps and integrates external C and C++ libraries.
*   **Performance Boost:** Significantly improves the speed of Python code execution.
*   **Large User Base:** Thousands of libraries and tools use Cython.
*   **Cross-Platform Compilation:** "Generate once, compile everywhere" C code generation that allows for reproducible performance results and testing
*   **Adaptability**: C compile time adaptation to the target platform and Python version
*   **Support for C-API implementations**: Seamless integration with C/C++ code
*   **Mature and Stable:** Benefit from over two decades of development, bug fixes, and optimization.

## Installation

To install Cython, ensure you have a C compiler and run the following command:

```bash
pip install Cython
```

For detailed installation instructions, see the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive [Apache License](https://github.com/cython/cython/blob/master/LICENSE.txt).

## Contributing

We welcome contributions!  Get started by reviewing the [contributing guidelines](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Differences from other Python Compilers

Cython stands out from other Python compilers due to its strong Python compatibility, seamless C/C++ integration, and extensive optimization capabilities.  Cython provides:

*   fast, efficient and highly compliant support for almost all Python language features, including dynamic features and introspection
*   full runtime compatibility with all still-in-use and future versions of CPython
*   "generate once, compile everywhere" C code generation that allows for reproducible performance results and testing
*   C compile time adaptation to the target platform and Python version
*   support for other C-API implementations, including PyPy and Pyston
*   seamless integration with C/C++ code
*   broad support for manual optimisation and tuning down to the C level
*   a large user base with thousands of libraries, packages and tools
*   more than two decades of bug fixing and static code optimisations

## Project Support

Support the Cython project through:

*   [Github Sponsors](https://github.com/users/scoder/sponsorship)
*   [Tidelift](https://tidelift.com/subscription/pkg/pypi-cython)

## Additional Resources

*   [Official Website](https://cython.org/)
*   [Documentation](https://docs.cython.org/)
*   [GitHub Repository](https://github.com/cython/cython)
*   [Wiki](https://github.com/cython/cython/wiki)