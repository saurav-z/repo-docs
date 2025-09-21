# ShivyC: A C Compiler Built in Python

**ShivyC is a hobby C compiler written in Python, designed to compile a subset of the C11 standard.**  ([Original Repo](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC aims to generate efficient binaries and provides helpful compile-time error messages.

**Key Features:**

*   **C11 Standard Subset:** Supports a portion of the C11 standard.
*   **Python-Based:** Entirely written in Python 3, making it accessible and easy to modify.
*   **x86-64 Assembly Generation:** Generates Intel-format x86-64 assembly code.
*   **Optimizations:** Includes optimizations during the compilation process.
*   **Error Messages:** Provides clear and helpful compile-time error messages.
*   **Register Allocation:** Employs George and Appel’s iterated register coalescing algorithm for register allocation.

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
    $ vim hello.c
    $ cat hello.c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  Compile and run:

    ```bash
    $ shivyc hello.c
    $ ./out
    hello, world!
    ```

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Using Docker (for non-Linux environments)

The [`docker/`](docker/) directory provides a Dockerfile to set up an x86-64 Linux Ubuntu environment.

1.  Clone the repository and build the Docker image:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    docker/shell
    ```

2.  Compile and test within the Docker environment:

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives. Implemented in `lexer.py` and `preproc.py`.
*   **Lexer:** Transforms source code into tokens. Implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`.
*   **Parser:** Uses recursive descent to create a parse tree. Implemented in `parser/*.py` and `tree/*.py`.
*   **IL Generation:** Converts the parse tree into a custom intermediate language (IL). Implemented in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`.
*   **ASM Generation:** Translates IL commands into x86-64 assembly code. Implemented in `asm_gen.py`, and `il_cmds/*.py`. Register allocation utilizes George and Appel’s algorithm.

## Contributing

This project is no longer under active development.

*   For questions, please use Github Issues.
*   To suggest practical improvements, please open an Issue.

## References

*   [ShivC (previous version)](https://github.com/ShivamSarodia/ShivC)
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)