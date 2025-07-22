# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, written in Python 3, that aims to support a subset of the C11 standard, generating efficient binaries with helpful compile-time error messages.**  [See the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:** Supports a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages.
*   **Written in Python:** Entirely implemented in Python 3.
*   **Optimizations:** Includes several optimization techniques.

## Quickstart

### x86-64 Linux

ShivyC requires Python 3.6 or later. Assembly and linking are done using GNU binutils and glibc.

**Installation:**

```bash
pip3 install shivyc
```

**Example Usage:**

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

**Running Tests:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Other Architectures (Using Docker)

A Dockerfile is provided for a pre-configured x86-64 Linux Ubuntu environment.

**Usage:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

This will open a shell with ShivyC installed, allowing you to:

```bash
shivyc any_c_file.c  # Compile a C file
python3 -m unittest discover  # Run tests
```

## Implementation Overview

ShivyC's compilation process includes these main stages:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the C code (primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Uses recursive descent techniques to build a parse tree (in `parser/*.py` using nodes defined in `tree/*.py`).
*   **IL Generation:** Transforms the parse tree into a custom intermediate language (IL), with commands defined in `il_cmds/*.py` and IL generation logic in `il_gen.py` and `tree/*.py`.
*   **ASM Generation:** Converts the IL to Intel-format x86-64 assembly code, using George and Appel's iterated register coalescing for register allocation (`asm_gen.py` and `il_cmds/*.py`).

## Contributing

This project is no longer under active development. Contributions are welcome.  Feel free to ask questions via Github Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler this project is based on.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf