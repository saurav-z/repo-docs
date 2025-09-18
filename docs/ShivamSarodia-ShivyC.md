# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built with Python, offering a unique perspective on compiler design and supporting a subset of the C11 standard.** Explore its capabilities and learn about compiler construction. Find the original repo [here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **Subset of C11 Standard:** Implements a portion of the C11 standard, enabling you to compile and run C code.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries, with optimizations included.
*   **Helpful Error Messages:** Provides informative compile-time error messages to aid in debugging.
*   **Written in Python:** Offers a Python-based implementation, making it accessible to Python developers interested in compiler design.
*   **Docker Support:** Provides a Docker environment for easy setup and use on various platforms.

## Quickstart

### Prerequisites
*   Python 3.6 or later
*   GNU binutils
*   glibc

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Example Usage

Create a `hello.c` file:

```c
$ vim hello.c
$ cat hello.c

#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```

Compile and run:

```bash
shivyc hello.c
./out
```

### Running Tests

1.  Clone the repository:
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```
2.  Run the tests:
    ```bash
    python3 -m unittest discover
    ```

### Using Docker

For cross-platform compatibility, use Docker:

1.  Clone the repository:
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```
2.  Enter the Docker environment:
    ```bash
    docker/shell
    ```

    Inside the Docker shell, compile and test your C files. Any local changes in the ShivyC directory will reflect in the Docker environment.

## Implementation Overview

ShivyC's architecture consists of the following stages:

*   **Preprocessor:**  Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:**  Transforms source code into tokens (implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:**  Uses recursive descent parsing to create a parse tree (in `parser/*.py` and `tree/*.py`).
*   **Intermediate Language (IL) Generation:**  Generates a custom IL from the parse tree (in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **Assembly (ASM) Generation:**  Converts IL commands into x86-64 assembly (in `asm_gen.py`, `il_cmds/*.py`). ShivyC uses George and Appel’s iterated register coalescing algorithm for register allocation.

## Contributing

This project is not actively maintained.  If you have questions or suggestions, please:

*   **Issues:**  Use GitHub Issues for any questions or suggestions.
*   **Feature Ideas:** Create issues to propose new features.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler that ShivyC is based on.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf