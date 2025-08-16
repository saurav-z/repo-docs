# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built from the ground up in Python, offering a unique learning experience and a functional C compiler.**  [Explore the original repository](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Subset:** Supports a functional subset of the C11 standard.
*   **Python-Based:** Written entirely in Python 3, making it accessible and easy to understand.
*   **Optimized Binaries:** Generates reasonably efficient x86-64 binaries with some optimizations.
*   **Helpful Error Messages:** Provides informative compile-time error messages for easier debugging.
*   **Educational:** An excellent resource for learning about compiler design and implementation.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Example Usage

1.  Create a simple "hello, world!" program (e.g., `hello.c`):

    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  Compile the program:

    ```bash
    shivyc hello.c
    ```

3.  Run the compiled executable:

    ```bash
    ./out
    ```

### Testing

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the tests:

    ```bash
    python3 -m unittest discover
    ```

### Docker Setup (for non-Linux Users)

For convenience, especially for those not running Linux, use the provided Dockerfile to set up an x86-64 Linux Ubuntu environment.

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker setup:

    ```bash
    docker/shell
    ```

    This will open a shell within a ShivyC-ready environment.  You can then compile and test as described above within the Docker container.

## Implementation Overview

ShivyC's compiler architecture is broken down into several stages:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Converts source code into tokens (primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Uses recursive descent techniques to build a parse tree (in `parser/*.py` and `tree/*.py`).
*   **Intermediate Language (IL) Generation:** Translates the parse tree into a custom IL (in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **Assembly (ASM) Generation:** Converts the IL into x86-64 assembly code, including register allocation using the George and Appel's iterated register coalescing algorithm (in `asm_gen.py` and `il_cmds/*.py`).

## Contributing

This project is no longer under active development. However:

*   **Questions:**  Ask questions using Github Issues.
*   **Suggestions:**  Suggest practical improvements via Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler this project is based on.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf