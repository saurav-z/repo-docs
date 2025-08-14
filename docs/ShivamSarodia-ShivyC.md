# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built in Python 3 that brings a subset of the C11 standard to life.**  Check out the original project on GitHub: [ShivamSarodia/ShivyC](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** Implements a portion of the C11 standard.
*   **Python-Based:** Written entirely in Python 3, making it accessible and easy to understand.
*   **Efficient Binaries:** Generates reasonably optimized x86-64 binaries.
*   **Compile-Time Error Messages:** Provides helpful error messages to aid in debugging.
*   **Includes:** Includes the basic features of a preprocessor

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Compile and Run a Simple Program

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

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker Support

For users not running Linux, a Dockerfile is provided in the [`docker/`](docker/) directory. This sets up an x86-64 Linux Ubuntu environment.

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker shell:

    ```bash
    docker/shell
    ```

3.  Inside the Docker environment:

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

    The Docker environment updates live with any changes made in your local ShivyC directory.

## Implementation Overview

ShivyC's architecture is divided into the following stages:

*   **Preprocessor:** Handles comments and `#include` directives. Implemented in [`lexer.py`](shivyc/lexer.py) and [`preproc.py`](shivyc/lexer.py).
*   **Lexer:** Converts source code into tokens. Implemented in [`lexer.py`](shivyc/lexer.py), with token definitions in [`tokens.py`](shivyc/tokens.py) and token kinds in [`token_kinds.py`](shivyc/token_kinds.py).
*   **Parser:** Uses recursive descent to create a parse tree. Implemented in [`parser/*.py`](shivyc/parser/) and utilizes tree nodes defined in [`tree/*.py`](shivyc/tree/).
*   **IL Generation:** Transforms the parse tree into a custom intermediate language (IL). IL commands are in [`il_cmds/*.py`](shivyc/il_cmds/), with objects in [`il_gen.py`](shivyc/il_gen.py), and most of the code in the `make_code` function in [`tree/*.py`](shivyc/tree/).
*   **ASM Generation:** Converts IL commands into x86-64 assembly.  Uses George and Appel’s iterated register coalescing algorithm for register allocation. General functionality is in [`asm_gen.py`](shivyc/asm_gen.py), with the bulk of the code in the `make_asm` function in [`il_cmds/*.py`](shivyc/il_cmds/).

## Contributing

Please note that this project is no longer under active development.

*   For questions, use GitHub Issues.
*   For suggestions, create an Issue.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf