# ShivyC: A C Compiler Written in Python

**ShivyC** is a hobby C compiler, built from scratch in Python 3, aiming to support a subset of the C11 standard. Check out the [original repo](https://github.com/ShivamSarodia/ShivyC) for more details!

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC aims to support a significant portion of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Optimizations:** Includes various optimizations for improved performance.
*   **Helpful Error Messages:** Provides clear and informative compile-time error messages.
*   **Written in Python:** Leverages the power and flexibility of Python 3 for compiler development.

## Quickstart Guide

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example Usage

1.  Create a C file (e.g., `hello.c`):

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

### Testing

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run tests:

    ```bash
    python3 -m unittest discover
    ```

### Docker for Other Architectures

For those not running Linux, use the provided Dockerfile:

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Enter the Docker environment:

    ```bash
    docker/shell
    ```

    Within the Docker environment, compile with `shivyc any_c_file.c` and test with `python3 -m unittest discover`.

## Implementation Overview

### Preprocessor

*   Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).

### Lexer

*   Tokenizes the input source code (implemented primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).

### Parser

*   Employs recursive descent parsing techniques (implemented in `parser/*.py`) to create a parse tree of nodes (defined in `tree/*.py`).

### IL Generation (Intermediate Language)

*   Traverses the parse tree to generate a custom, flat IL (commands defined in `il_cmds/*.py`, objects in `il_gen.py`, with most IL generation within the `make_code` functions of the tree nodes in `tree/*.py`).

### ASM Generation (x86-64 Assembly)

*   Converts IL commands into Intel-format x86-64 assembly code (general functionality in `asm_gen.py`, with assembly generation mostly in the `make_asm` functions of IL commands in `il_cmds/*.py`).
*   Employs George and Appel's iterated register coalescing algorithm for register allocation.

## Contributing

This project is no longer under active development.

*   **Questions:** Use GitHub Issues.
*   **Suggestions:** Submit an Issue with perspectives for practical improvements.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler from which ShivyC was rewritten.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf