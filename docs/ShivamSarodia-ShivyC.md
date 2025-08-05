# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, written entirely in Python, that brings the C11 standard to life.** (See original repo: [ShivamSarodia/ShivyC](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC supports a subset of the C11 standard and generates reasonably efficient x86-64 binaries with optimizations. It also provides helpful compile-time error messages.

## Key Features

*   **C11 Subset Support:** Implements a portion of the C11 standard.
*   **Python-Based:** Written entirely in Python 3.
*   **Optimizations:** Includes optimizations for generated binaries.
*   **Error Messages:** Generates helpful compile-time error messages.
*   **x86-64 Architecture:** Generates assembly code for the x86-64 architecture.
*   **Uses GNU binutils and glibc:** For assembling and linking.

## Quickstart

### Prerequisites
* Python 3.6 or later
* GNU binutils and glibc (usually pre-installed on Linux systems)

### Installation

```bash
pip3 install shivyc
```

### Example: Compile and Run "Hello, World!"

1.  Create `hello.c`:

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

    This will compile `hello.c` and create an executable named `out`, which will print "hello, world!" to the console.

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Using Docker for Other Architectures

For ease of use, particularly if you're not running Linux, you can use the provided Dockerfile.

1.  Build and run the Docker environment:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    docker/shell
    ```

2.  Within the Docker shell:

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

    Changes made to your local ShivyC directory will be reflected live in the Docker environment.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Transforms the source code into tokens (implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:** Uses recursive descent to create a parse tree from the tokens (implemented in `parser/*.py` and `tree/*.py`).
*   **IL Generation:** Generates a custom intermediate language (IL) by traversing the parse tree (implemented in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Converts the IL commands into x86-64 assembly code using George and Appel's iterated register coalescing algorithm (implemented in `asm_gen.py`, and `il_cmds/*.py`).

## Contributing

This project is no longer under active development. However:

*   **Questions:** Please ask questions via Github Issues.
*   **Suggestions:** Propose ideas for practical improvements via Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler written by the author, from which ShivyC was rewritten.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf