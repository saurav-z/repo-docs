# ShivyC: A C Compiler Written in Python

**ShivyC** is a hobby C compiler written in Python that allows you to compile a subset of the C11 standard. Check out the original repository [here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC implements a subset of the C11 standard.
*   **Python-Based:** Built entirely in Python 3.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Compile-Time Error Messages:** Provides helpful error messages to aid in debugging.
*   **Optimizations:** Includes some optimizations to improve code performance.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example Usage

1.  Create a `hello.c` file:

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

    Output:
    ```
    hello, world!
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

### Docker Quickstart (for other architectures or environments)

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker setup:

    ```bash
    docker/shell
    ```

    Within the Docker shell, you can compile and test:

    ```bash
    shivyc any_c_file.c  # compile a C file
    python3 -m unittest discover  # run tests
    ```

    The Docker environment automatically reflects local changes to ShivyC.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Transforms source code into tokens (implemented primarily in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:** Uses recursive descent parsing to create a parse tree (in `parser/*.py` and `tree/*.py`).
*   **IL Generation:** Generates a custom Intermediate Language (IL) from the parse tree (in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code, including register allocation using George and Appel’s iterated register coalescing algorithm (in `asm_gen.py`, `il_cmds/*.py`).

## Contributing

*   This project is no longer under active development, but you can still submit issues or provide perspectives.
*   If you have questions, create a Github Issue.
*   If you have ideas, open an Issue to share.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)