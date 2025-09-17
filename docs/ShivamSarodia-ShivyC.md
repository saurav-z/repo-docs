# ShivyC: A Hobby C Compiler in Python

**ShivyC is a Python-based hobby C compiler that brings C11 support and efficient binaries to life.** Check out the original project on [GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** Compiles a significant portion of the C11 standard.
*   **Efficient Binaries:** Generates reasonably optimized x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages to aid debugging.
*   **Written in Python:** Leverages the versatility and readability of Python 3.

## Quickstart Guide

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Compile and Run a C Program

1.  Create a `hello.c` file:

    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```
2.  Compile and run:

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

### Docker Environment

For easy setup on any architecture, use the provided Dockerfile:

1.  Navigate to the `docker/` directory.
2.  Build and run the Docker container:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    docker/shell
    ```
3.  Inside the container, use ShivyC:

    ```bash
    shivyc any_c_file.c           # Compile a C file
    python3 -m unittest discover  # Run tests
    ```

## Implementation Overview

*   **Preprocessor:** Handles comments and includes (lexer.py, preproc.py).
*   **Lexer:** Tokenizes the C code (lexer.py, tokens.py, token_kinds.py).
*   **Parser:** Employs recursive descent parsing (parser/\*.py, tree/\*.py).
*   **IL Generation:** Converts parsed code into a custom intermediate language (il\_cmds/\*.py, il\_gen.py, tree/\*.py).
*   **ASM Generation:** Converts the IL to x86-64 assembly, including register allocation (asm\_gen.py, il\_cmds/\*.py).

## Contributing

This project is not under active development, but issues and PRs are welcome.

## References

*   [ShivyC on GitHub](https://github.com/ShivamSarodia/ShivC) - The previous C compiler by the same author.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)