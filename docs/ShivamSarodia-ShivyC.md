# ShivyC: A C Compiler in Python

**ShivyC is a hobby C compiler written in Python, supporting a subset of the C11 standard and designed to generate reasonably efficient binaries with helpful compile-time error messages.**  [View the source on GitHub](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **Subset of C11 Standard:** ShivyC supports a significant portion of the C11 standard.
*   **Python-Based:** Built entirely in Python 3.
*   **Efficient Binaries:** Generates reasonably optimized x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages to aid debugging.
*   **Includes optimizations** ShivyC includes optimizations.
*   **x86-64 Linux Support:**  Designed to work seamlessly on x86-64 Linux systems with GNU binutils and glibc.
*   **Docker Support:** Includes a Dockerfile for easy setup in a Linux environment.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example Usage

1.  **Create a `hello.c` file:**

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

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Other Architectures (Docker)

For convenience, a Dockerfile is provided to set up an x86-64 Linux Ubuntu environment.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  **Run the Docker shell:**

    ```bash
    docker/shell
    ```

    This will open a shell with ShivyC installed and ready to use.

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

## Implementation Overview

ShivyC's compilation process consists of the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the input source code (primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Uses recursive descent techniques to build a parse tree (in `parser/*.py`, creating nodes defined in `tree/*.py`).
*   **IL Generation:** Converts the parse tree into a custom intermediate language (IL) (in `il_cmds/*.py`, `il_gen.py` and `tree/*.py`).
*   **ASM Generation:** Translates the IL into x86-64 assembly code, including register allocation using George and Appel’s iterated register coalescing algorithm (in `asm_gen.py` and `il_cmds/*.py`).

## Contributing

This project is no longer under active development. However, contributions and feedback are welcome!

*   **Questions:**  Use GitHub Issues to ask questions.
*   **Suggestions:** Submit an Issue with suggestions for practically helpful changes.

## References

*   [ShivC (Original Compiler)](https://github.com/ShivamSarodia/ShivC) - The predecessor to ShivyC.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf