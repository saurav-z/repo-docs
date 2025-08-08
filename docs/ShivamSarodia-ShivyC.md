# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built with Python that aims to provide a learning experience in compiler design and the C language.** [See the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC supports a portion of the C11 standard.
*   **Python-Based:** Built entirely in Python 3, making it accessible and easier to understand.
*   **Optimizations:** Includes basic optimizations to generate reasonably efficient binaries.
*   **Clear Error Messages:** Provides helpful compile-time error messages to aid debugging.
*   **x86-64 Assembly Generation:** Generates Intel-format x86-64 assembly code.
*   **Iterated Register Coalescing:** Employs the George and Appel's algorithm for register allocation.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (for assembling and linking)

### Installation

```bash
pip3 install shivyc
```

### Example Usage

1.  Create a `hello.c` file:

    ```c
    $ vim hello.c
    $ cat hello.c

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

2.  Run tests:

    ```bash
    python3 -m unittest discover
    ```

### Docker Environment

For users not on Linux, a Dockerfile is provided for an x86-64 Linux Ubuntu environment.

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Enter the Docker shell:

    ```bash
    docker/shell
    ```

    Within the Docker shell:

    *   Compile: `shivyc any_c_file.c`
    *   Run tests: `python3 -m unittest discover`

## Implementation Overview

*   **Preprocessor:**  Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the input source code (implemented primarily in `lexer.py` and with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Uses recursive descent techniques to parse the code into a parse tree (implemented in `parser/*.py`, creating tree nodes defined in `tree/*.py`).
*   **IL Generation:** Generates a custom intermediate language (IL) from the parse tree (commands in `il_cmds/*.py`, objects in `il_gen.py`, and `make_code` functions in `tree/*.py`).
*   **ASM Generation:** Transforms IL commands into x86-64 assembly code. Includes register allocation using George and Appel's algorithm (general functionality in `asm_gen.py`, and  `make_asm` functions in `il_cmds/*.py`).

## Contributing

*   This project is no longer under active development.
*   For questions, use Github Issues.
*   For suggestions on how ShivyC can be practically helpful, create an Issue.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler from which ShivyC was rewritten.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf