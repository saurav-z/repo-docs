# ShivyC: A C Compiler Written in Python

**ShivyC** is a hobby C compiler written in Python, offering a glimpse into compiler design and supporting a subset of the C11 standard.  [Explore the ShivyC repository on GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** Compiles a portion of the C11 standard.
*   **Python-Based:** Written entirely in Python 3.
*   **x86-64 Assembly Generation:** Generates x86-64 assembly code.
*   **Optimizations:** Includes basic optimizations for more efficient binaries.
*   **Compile-Time Error Messages:** Provides helpful error messages to aid debugging.
*   **Includes a limited preprocessor**

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux systems)

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

2.  Compile with ShivyC:

    ```bash
    shivyc hello.c
    ```

3.  Run the executable:

    ```bash
    ./out
    ```

### Running Tests

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

For those not on Linux, or for a more isolated environment, use the provided Dockerfile:

1.  Clone the repository (if you haven't already):

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Start the Docker shell:

    ```bash
    docker/shell
    ```

    This will launch a shell with ShivyC pre-installed. You can then compile and test C files within the Docker environment.  Changes to files in your local ShivyC directory will be reflected live in the Docker environment.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the input code (primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Uses recursive descent to parse the tokens and create a parse tree (in `parser/*.py` and `tree/*.py`).
*   **Intermediate Language (IL) Generation:** Transforms the parse tree into a custom IL (in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **Assembly (ASM) Generation:** Converts the IL commands into Intel-format x86-64 assembly code, including register allocation using the George and Appel's iterated register coalescing algorithm (in `asm_gen.py`, `il_cmds/*.py`).

## Contributing

While active development is limited, questions and perspectives are welcome via Github Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler written by the author.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf