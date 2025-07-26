# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, providing a unique opportunity to learn about compilers and programming language internals.**  [View the source on GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC is a hobby C compiler written in Python 3 that supports a subset of the C11 standard and generates reasonably efficient binaries, including some optimizations. It also provides helpful compile-time error messages.

## Key Features

*   **Written in Python:**  Provides an accessible entry point for understanding compiler design.
*   **Supports a Subset of C11:**  Compiles a functional subset of C code.
*   **Generates x86-64 Binaries:** Produces native executables for Linux.
*   **Includes Optimizations:** Offers basic optimizations for performance.
*   **Provides Clear Error Messages:**  Aids in debugging and understanding compilation issues.
*   **Docker Support:** Includes a Dockerfile for easy setup and use on various platforms.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example: Compile and Run a "Hello, World!" Program

1.  Create a `hello.c` file:

    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  Compile the C code:

    ```bash
    shivyc hello.c
    ```

3.  Run the compiled executable:

    ```bash
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

For convenience, especially for non-Linux environments, use Docker:

1.  Clone the repository (if you haven't already):

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker setup:

    ```bash
    docker/shell
    ```

    This provides a shell with ShivyC installed.  You can then compile and test files as described above, using `shivyc` and `python3 -m unittest discover`. Changes in your local ShivyC directory are reflected immediately within the Docker environment.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the input source code (primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Uses recursive descent to create a parse tree (in `parser/*.py`, using tree nodes defined in `tree/*.py`).
*   **IL Generation:**  Transforms the parse tree into a custom intermediate language (IL) (commands in `il_cmds/*.py`, generation logic primarily in `tree/*.py`'s `make_code` functions).
*   **ASM Generation:** Converts the IL into x86-64 assembly code (in `asm_gen.py` and `il_cmds/*.py`'s `make_asm` functions), including register allocation using George and Appel's iterated register coalescing algorithm.

## Contributing

While active development is limited, questions via GitHub Issues are welcome.  Also, feel free to open issues if you have ideas about making ShivyC practically helpful.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler ShivyC was rewritten from.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf