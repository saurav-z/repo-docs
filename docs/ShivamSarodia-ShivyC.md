# ShivyC: A C Compiler Written in Python 🐍

**ShivyC is a hobby C compiler built in Python, aiming to compile a subset of the C11 standard and generate efficient binaries.** ([Original Repository](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** Implements a portion of the C11 standard.
*   **Optimized Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** Provides clear compile-time error messages.
*   **Written in Python:** Built entirely using Python 3.
*   **Includes a simple example of a trie implementation.**

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example: Compile and Run "Hello, World!"

1.  Create `hello.c`:

    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  Compile:

    ```bash
    shivyc hello.c
    ```

3.  Run:

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

### Using Docker (for non-Linux environments)

The `docker/` directory contains a Dockerfile for setting up an x86-64 Linux Ubuntu environment with ShivyC.

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker shell:

    ```bash
    docker/shell
    ```

    This provides a shell with ShivyC pre-installed. You can then compile files with `shivyc any_c_file.c` and run tests with `python3 -m unittest discover`.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (lexer.py, preproc.py).
*   **Lexer:** Converts source code into tokens (lexer.py, tokens.py, token_kinds.py).
*   **Parser:** Utilizes recursive descent parsing to create a parse tree (parser/\*.py, tree/\*.py).
*   **IL Generation:** Transforms the parse tree into a custom intermediate language (IL) (il_cmds/\*.py, il_gen.py, tree/\*.py).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code, using George and Appel’s iterated register coalescing (asm_gen.py, il_cmds/\*.py).

## Contributing

*   This project is no longer under active development.
*   Questions can be asked via GitHub Issues.
*   Suggestions on making ShivyC helpful are welcome via Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC)
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf