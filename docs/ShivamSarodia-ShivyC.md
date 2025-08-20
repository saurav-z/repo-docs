# ShivyC: A C Compiler in Python

**ShivyC is a hobby C compiler built in Python that brings a subset of the C11 standard to life, allowing you to compile and run C code with helpful error messages.** Check out the original repository for more information: [https://github.com/ShivamSarodia/ShivyC](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:** ShivyC compiles a subset of the C11 standard, allowing you to experiment with a core set of C features.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries, including optimizations to improve performance.
*   **Helpful Error Messages:** Provides informative compile-time error messages to assist in debugging.
*   **Written in Python:** Built entirely in Python 3, making it accessible and easy to understand.
*   **x86-64 Linux Support:** Compiles and runs on x86-64 Linux systems, leveraging GNU binutils and glibc for assembly and linking.

## Quickstart

### x86-64 Linux

**Prerequisites:** Python 3.6 or later, GNU binutils, and glibc (typically pre-installed on Linux systems).

1.  **Install:**
    ```bash
    pip3 install shivyc
    ```
2.  **Create, Compile, and Run a "Hello, World!" Program:**

    ```c
    $ vim hello.c
    $ cat hello.c

    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }

    $ shivyc hello.c
    $ ./out
    hello, world!
    ```

3.  **Run Tests:**
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    python3 -m unittest discover
    ```

### Other Architectures (Docker)

For convenience, a Dockerfile is provided in the [`docker/`](docker/) directory to set up an x86-64 Linux Ubuntu environment.

1.  **Clone the Repository and Build/Enter Docker:**
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    docker/shell
    ```
2.  **Compile and Run Inside Docker:**

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

    *Note: The Docker ShivyC executable updates live with local changes in your ShivyC directory.*

## Implementation Overview

*   **Preprocessor:** Limited preprocessor functionality includes comment removal and `#include` directive expansion (lexer.py and preproc.py).
*   **Lexer:** Implemented in lexer.py, with token definitions in tokens.py and token kinds in token\_kinds.py.
*   **Parser:** Uses recursive descent techniques (parser/\*.py) to create a parse tree (tree/\*.py).
*   **IL Generation:** Generates a custom intermediate language (IL) by traversing the parse tree (il\_cmds/\*.py, il\_gen.py, and tree/\*.py).
*   **ASM Generation:** Converts IL commands into Intel-format x86-64 assembly code, with register allocation using George and Appel's iterated register coalescing algorithm (asm\_gen.py and il\_cmds/\*.py).

## Contributing

This project is no longer under active development. However, you can still contribute!

*   **Questions:** Ask questions via GitHub Issues.
*   **Suggestions:** Suggest improvements via GitHub Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler which this project is a rewrite of.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf