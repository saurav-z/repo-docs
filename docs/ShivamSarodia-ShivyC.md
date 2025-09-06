# ShivyC: A C Compiler in Python

**ShivyC is a hobby C compiler written in Python, offering a unique perspective on compiler design.** [View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC is a project focused on creating a C compiler in Python that supports a subset of the C11 standard. It generates reasonably efficient x86-64 binaries and includes optimizations, as well as helpful compile-time error messages.

## Key Features

*   **Written in Python:** Explore compiler internals using a modern, accessible language.
*   **C11 Subset Support:** Compiles a portion of the C11 standard.
*   **Optimizations:** Includes optimizations for improved binary performance.
*   **Clear Error Messages:** Provides helpful feedback during compilation.
*   **x86-64 Binary Generation:** Targets the x86-64 architecture for Linux.
*   **Docker Support:** Easy setup via Docker for cross-platform use.

## Quickstart

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

2.  Compile and Run:

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

### Docker Quickstart

For environments other than Linux, the `docker/` directory provides a Dockerfile to set up a Linux environment.

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```
This opens a shell with ShivyC installed:
```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```
The Docker ShivyC executable will update live with any changes made in your local ShivyC directory.

## Implementation Overview

ShivyC's architecture follows a typical compiler structure:

*   **Preprocessor:** Handles comments and `#include` directives (`lexer.py`, `preproc.py`).
*   **Lexer:** Transforms source code into tokens (`lexer.py`, `tokens.py`, `token_kinds.py`).
*   **Parser:** Employs recursive descent to create a parse tree (`parser/*.py`, `tree/*.py`).
*   **IL Generation:** Converts the parse tree into a custom intermediate language (`il_cmds/*.py`, `il_gen.py`, `tree/*.py`).
*   **ASM Generation:** Translates IL commands into x86-64 assembly code, including register allocation using George and Appel’s iterated register coalescing algorithm (`asm_gen.py`, `il_cmds/*.py`).

## Contributing

While active development is limited, questions and suggestions are welcome via GitHub Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC): The original compiler ShivyC was based on.
*   C11 Specification: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI: https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel): https://www.cs.purdue.edu/homes/hosking/502/george.pdf