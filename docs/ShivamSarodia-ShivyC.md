# ShivyC: A C Compiler Built in Python

**ShivyC is a hobby C compiler, written entirely in Python, that aims to bring you closer to understanding the inner workings of a compiler while supporting a subset of the C11 standard.**  [View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC provides a practical learning experience for compiler design, demonstrating how C code is transformed into executable machine code.

## Key Features

*   **C11 Standard Subset:** Supports a portion of the C11 standard.
*   **Python-Based:**  Written in Python 3, making it accessible and easier to understand.
*   **X86-64 Binary Generation:**  Generates reasonably efficient x86-64 binaries.
*   **Compile-Time Error Messages:** Provides helpful error messages to aid in debugging.
*   **Includes Optimizations:** Incorporates optimizations to improve generated code.

## Quickstart Guide

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example: Compile and Run "Hello, World!"

1.  **Create `hello.c`:**

    ```c
    $ vim hello.c
    $ cat hello.c

    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  **Compile and Run:**

    ```bash
    shivyc hello.c
    ./out
    ```

### Testing

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  **Run Tests:**

    ```bash
    python3 -m unittest discover
    ```

### Docker for Other Architectures

For users not running Linux, a Dockerfile is provided.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  **Run Docker:**

    ```bash
    docker/shell
    ```

    This opens a shell with ShivyC installed.  Use it to compile and test your C files.

## Implementation Overview

*   **Preprocessor:**  Handles comments and `#include` directives. (Implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Transforms source code into tokens. (Implemented primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Uses recursive descent parsing to build an abstract syntax tree. (Implemented in `parser/*.py` and `tree/*.py`).
*   **IL Generation:**  Generates a custom intermediate language. (Implemented in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code.  (Implemented in `asm_gen.py` and `il_cmds/*.py`). Utilizes George and Appel's iterated register coalescing algorithm.

## Contributing

This project is no longer under active development.  However, you can still contribute:

*   **Questions:** Use Github Issues to ask about ShivyC.
*   **Suggestions:** Submit Issues with ideas on how ShivyC can be useful.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - Original compiler project.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf