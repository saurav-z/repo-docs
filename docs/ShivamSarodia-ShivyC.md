# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler that lets you compile C code into efficient binaries, all thanks to Python.** ([View on GitHub](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC is a hobby C compiler built in Python 3, supporting a subset of the C11 standard. It generates optimized binaries and provides helpful compile-time error messages.

**Key Features:**

*   **C11 Subset:** Supports a portion of the C11 standard.
*   **Optimized Binaries:** Generates reasonably efficient executable files.
*   **Clear Error Messages:** Provides helpful compile-time error messages for easier debugging.
*   **Written in Python:** Leverages the versatility of Python for compiler development.

## Getting Started

### Installation

1.  **Prerequisites:** Python 3.6 or later, GNU binutils, and glibc.
2.  **Install ShivyC:**

    ```bash
    pip3 install shivyc
    ```

### Example: Hello, World!

1.  **Create `hello.c`:**

    ```c
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

For users not running Linux, a Dockerfile is provided in the `docker/` directory.

1.  **Clone the Repository (if you haven't already):**

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  **Run the Docker Shell:**

    ```bash
    docker/shell
    ```

    This provides a shell with ShivyC installed. Use `shivyc any_c_file.c` to compile and `python3 -m unittest discover` to run tests within the Docker environment.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the input code (primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Uses recursive descent parsing to build a parse tree (`parser/*.py` and `tree/*.py`).
*   **IL Generation:** Generates a custom intermediate language (IL) from the parse tree (`il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code, including register allocation using George and Appel’s iterated register coalescing algorithm (`asm_gen.py` and `il_cmds/*.py`).

## Contributing

This project is no longer under active development. However:

*   **Questions:** Open issues on GitHub.
*   **Suggestions:** Propose improvements through GitHub issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler, ShivC, on which ShivyC is based.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf