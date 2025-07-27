# ShivyC: A Hobby C Compiler in Python

**ShivyC is a C compiler written in Python that translates a subset of the C11 standard to x86-64 assembly code, complete with optimizations.** Dive in and explore the inner workings of a compiler with this educational and practical project. [Learn more on the original repository](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:** Implements a subset of the C11 standard.
*   **Optimized Binaries:** Generates reasonably efficient x86-64 assembly code.
*   **Informative Error Messages:** Provides helpful compile-time error messages to aid debugging.
*   **Written in Python:** Leverages Python 3 for compiler implementation.
*   **Includes a Docker Environment:** Provides a Dockerfile to set up an x86-64 Linux Ubuntu environment for easy use.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically already installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example: Compile and Run a "Hello, World!" program

1.  Create a `hello.c` file:

    ```c
    $ vim hello.c
    $ cat hello.c

    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  Compile with ShivyC:

    ```bash
    shivyc hello.c
    ```

3.  Run the compiled program:

    ```bash
    ./out
    hello, world!
    ```

### Run Tests

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run tests:

    ```bash
    python3 -m unittest discover
    ```

### Using Docker

For users who are not running Linux, you can use the provided Dockerfile:

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Start the Docker environment:

    ```bash
    docker/shell
    ```

3.  Compile a C file:

    ```bash
    shivyc any_c_file.c
    ```

4.  Run tests:

    ```bash
    python3 -m unittest discover
    ```

## Implementation Overview

ShivyC's architecture is comprised of several key stages:

*   **Preprocessor:** Handles comments and `#include` directives (lexer.py, preproc.py).
*   **Lexer:** Breaks down the source code into tokens (lexer.py, tokens.py, token\_kinds.py).
*   **Parser:** Uses recursive descent to create a parse tree (parser/\*.py, tree/\*.py).
*   **Intermediate Language (IL) Generation:** Transforms the parse tree into a custom IL (il\_cmds/\*.py, il\_gen.py, tree/\*.py).
*   **Assembly Generation:** Converts IL commands into Intel-format x86-64 assembly code, including register allocation using George and Appel’s iterated register coalescing algorithm (asm\_gen.py, il\_cmds/\*.py).

## Contributing

The project is no longer under active development, but contributions are welcome:

*   For questions, use Github Issues.
*   To suggest improvements, create an Issue.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler this is based on.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf