# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built in Python, enabling you to compile a subset of the C11 standard and generate efficient binaries.** (Check out the original repo: [https://github.com/ShivamSarodia/ShivyC](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** Compiles a portion of the C11 standard.
*   **Efficient Binaries:** Generates reasonably optimized x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages.
*   **Written in Python:** Leverages the Python programming language.
*   **Intermediate Language (IL):** Utilizes a custom IL for efficient compilation.
*   **Register Allocation:** Employs George and Appel's iterated register coalescing algorithm.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils
*   glibc

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

    This will compile `hello.c` and create an executable named `out`.

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

### Docker (for non-Linux users)

A Dockerfile is provided in the `docker/` directory for convenient compilation in a Linux environment:

1.  Clone the repository.
2.  Navigate to the repository's root directory: `cd ShivyC`
3.  Run the setup shell with:  `docker/shell`

This sets up a shell with ShivyC installed and ready to use.  Any local changes to your ShivyC directory are instantly reflected in the Docker environment.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Converts source code into tokens (`lexer.py`, `tokens.py`, `token_kinds.py`).
*   **Parser:** Uses recursive descent to create a parse tree (`parser/*.py`, `tree/*.py`).
*   **IL Generation:** Transforms the parse tree into a custom IL (`il_cmds/*.py`, `il_gen.py`, `tree/*.py`).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code, including register allocation (`asm_gen.py`, `il_cmds/*.py`).

## Contributing

This project is no longer under active development. However, if you have questions, create a GitHub issue.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC)
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf