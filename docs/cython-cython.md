# Cython: The Ultimate Python Compiler for Speed and Efficiency

**Cython empowers you to write C extensions for Python with ease, boosting performance and unlocking the power of C/C++ within your Python code.**  [Learn more at the original Cython repository](https://github.com/cython/cython).

## Key Features:

*   **Performance Optimization:** Translates Python code into highly efficient C/C++ code, significantly speeding up execution.
*   **Seamless C Integration:** Effortlessly calls C functions and integrates with C/C++ libraries.
*   **Fine-Grained Control:** Allows declaring C types for variables and class attributes for advanced tuning.
*   **Versatile Application:** Ideal for wrapping C libraries and building fast C modules to accelerate Python code.
*   **Broad Python Compatibility:** Supports almost all Python language features, ensuring compatibility with CPython and other implementations.
*   **Cross-Platform Support:** Generates C code adaptable to the target platform and Python version.
*   **Mature and Widely Used:** Backed by a large user base and over two decades of development, with over 70 million downloads per month on PyPI.

## Installation

To install Cython, simply run the following command (assuming you have a C compiler):

```bash
pip install Cython
```

For detailed installation instructions and troubleshooting, please refer to the [installation page](https://docs.cython.org/en/latest/src/quickstart/install.html).

## License

Cython is licensed under the permissive **Apache License**. The original Pyrex program, upon which Cython is based, was licensed "free of restrictions".

See the full license details in `LICENSE.txt <https://github.com/cython/cython/blob/master/LICENSE.txt>`.

## Contributing

Interested in contributing to the Cython project?  Get started with our helpful [contribution guide](https://github.com/cython/cython/blob/master/docs/CONTRIBUTING.rst).

## Cython vs. Other Python Compilers

Cython stands out among Python compilers due to its unique strengths:

*   **PyPy:** Offers JIT compilation, but with a non-CPython runtime and potential compatibility limitations.
*   **Numba:** Specializes in numerical code with JIT compilation, but has language support limitations.
*   **Pythran:**  Focuses on numerical computation, best utilized as a backend for NumPy code in Cython.
*   **mypyc:**  Provides static compilation, with support for PEP-484 typing, but sacrifices low-level optimization.
*   **Nuitka:** Offers static compilation with strong language compliance, but lacks low-level optimization capabilities.

**Cython's key advantages:**

*   **Exceptional Python Compatibility:** Supports almost all Python features.
*   **Full CPython Compatibility:** Works seamlessly with CPython versions.
*   **Reproducible Performance:** Enables consistent performance results with C code generation.
*   **C/C++ Integration:** Integrates easily with C/C++ code.
*   **Extensive Optimization Options:** Offers advanced manual tuning.
*   **Large Ecosystem:** Supported by a vast user community and extensive library support.

## About Pyrex (Foundation of Cython)

Cython is built upon the foundation of Pyrex, a language created by Greg Ewing. Pyrex aimed to simplify the creation of Python extension modules.