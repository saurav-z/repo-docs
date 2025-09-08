# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, written in Python 3, that lets you compile and run C code with helpful error messages.**

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

[View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC)

ShivyC is a C compiler project built as a hobby project, offering a glimpse into the inner workings of compilers. It's written in Python 3 and supports a subset of the C11 standard, generating reasonably efficient x86-64 binaries. ShivyC is designed to provide helpful compile-time error messages, making it a useful tool for learning and experimentation.

## Key Features

*   **C11 Subset Support:** Compiles a portion of the C11 standard.
*   **Python-Based:** Written entirely in Python 3, making it accessible and easy to understand for Python developers.
*   **x86-64 Binary Generation:** Creates executable binaries for x86-64 Linux systems.
*   **Optimizations:** Includes optimizations to generate more efficient code.
*   **Helpful Error Messages:** Provides informative compile-time error messages to aid in debugging.
*   **Docker Support:** Includes a Dockerfile for convenient use on various platforms.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example Usage

1.  Create a C file, such as `hello.c`:

    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  Compile and Run:

    ```bash
    shivyc hello.c
    ./out
    ```
    This will compile `hello.c` and generate an executable `out`. Running `./out` will print "hello, world!".

### Running Tests

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the tests:

    ```bash
    python3 -m unittest discover
    ```

### Using Docker (for non-Linux environments)

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker shell:

    ```bash
    docker/shell
    ```

    This will open a shell in a pre-configured environment with ShivyC ready to use.

    *   Compile a file: `shivyc any_c_file.c`
    *   Run tests: `python3 -m unittest discover`
    *   The Docker ShivyC executable will update live with any changes made in your local ShivyC directory.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives, using files between [`lexer.py`](shivyc/lexer.py) and [`preproc.py`](shivyc/lexer.py).
*   **Lexer:** Tokenizes the input source code, primarily using [`lexer.py`](shivyc/lexer.py) along with token definitions in [`tokens.py`](shivyc/tokens.py) and [`token_kinds.py`](shivyc/token_kinds.py).
*   **Parser:** Employs recursive descent techniques for parsing, located in [`parser/*.py`](shivyc/parser/), creating a parse tree from nodes defined in [`tree/*.py`](shivyc/tree/).
*   **IL Generation:** Transforms the parse tree into a custom Intermediate Language (IL), using commands defined in [`il_cmds/*.py`](shivyc/il_cmds/), objects in [`il_gen.py`](shivyc/il_gen.py), and the `make_code` functions in [`tree/*.py`](shivyc/tree/).
*   **ASM Generation:** Converts IL commands into Intel-format x86-64 assembly code, using register allocation with George and Appel's iterated register coalescing algorithm, with code in [`asm_gen.py`](shivyc/asm_gen.py) and `make_asm` functions in [`il_cmds/*.py`](shivyc/il_cmds/).

## Contributing

This project is no longer under active development. If you have any questions or ideas, please create an issue on GitHub.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC): The original C compiler project that ShivyC is based on.
*   C11 Specification: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI: https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel): https://www.cs.purdue.edu/homes/hosking/502/george.pdf