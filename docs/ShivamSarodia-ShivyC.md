# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, meticulously crafted in Python 3, offering a glimpse into the fascinating world of compiler design.** [Check out the original repository](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Subset:** Supports a portion of the C11 standard, enabling you to compile and run C code.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries, incorporating several optimization techniques.
*   **Helpful Error Messages:** Provides informative compile-time error messages to aid in debugging.
*   **Written in Python:**  A great resource for learning about compilers and programming in Python.

## Quickstart Guide

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

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

2.  Compile and Run:

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

### Docker Environment

For those not running Linux, a Dockerfile is provided to set up an x86-64 Ubuntu environment:

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker shell:

    ```bash
    docker/shell
    ```

    Now, you can use `shivyc` and run tests within the Docker environment.

## Implementation Overview

ShivyC's architecture is broken down into key components:

*   **Preprocessor:** Handles comments and `#include` directives (`lexer.py`, `preproc.py`).
*   **Lexer:**  Transforms source code into tokens (`lexer.py`, `tokens.py`, `token_kinds.py`).
*   **Parser:**  Uses recursive descent parsing to build a parse tree (`parser/*.py`, `tree/*.py`).
*   **IL Generation:** Creates an Intermediate Language representation from the parse tree (`il_cmds/*.py`, `il_gen.py`, `tree/*.py`).
*   **ASM Generation:** Converts the IL into x86-64 assembly code, using register allocation via George and Appel’s iterated register coalescing (`asm_gen.py`, `il_cmds/*.py`).

## Contributing

*   **Seeking answers:** Post questions on Github Issues.  Responses may be delayed.
*   **Suggestions welcome:**  Suggest improvements via Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler project.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf