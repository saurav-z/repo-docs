# ShivyC: A C Compiler in Python for Learning and Exploration

ShivyC is a hobby C compiler, written in Python, offering a glimpse into the inner workings of a compiler, supporting a subset of the C11 standard. Explore the source code on [GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **Written in Python:**  ShivyC is implemented entirely in Python 3, making it easier to understand and modify.
*   **C11 Subset:** Supports a functional subset of the C11 standard.
*   **Optimized Binaries:** Generates reasonably efficient x86-64 binaries with some optimizations.
*   **Clear Error Messages:** Provides helpful compile-time error messages to aid debugging.
*   **Educational:** A great tool for learning about compiler design and the C language.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example: "Hello, World!"

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

### Docker for Other Architectures

For users on non-Linux systems, use the provided Dockerfile:

1.  Clone the repository and navigate to the directory:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Start a shell in the Docker environment:

    ```bash
    docker/shell
    ```

    *   Inside the Docker shell, you can compile with `shivyc your_c_file.c` and run tests with `python3 -m unittest discover`.
    *   Changes made to the local ShivyC directory are reflected live within the Docker environment.

## Implementation Overview

ShivyC's compiler pipeline is broken down into the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in [`lexer.py`](shivyc/lexer.py) and [`preproc.py`](shivyc/lexer.py)).
*   **Lexer:** Converts source code into tokens (implemented in [`lexer.py`](shivyc/lexer.py), [`tokens.py`](shivyc/tokens.py), and [`token_kinds.py`](shivyc/token_kinds.py)).
*   **Parser:**  Employs recursive descent to construct a parse tree (implemented in [`parser/*.py`](shivyc/parser/) and [`tree/*.py`](shivyc/tree/)).
*   **Intermediate Language (IL) Generation:** Transforms the parse tree into a custom IL (implemented in [`il_cmds/*.py`](shivyc/il_cmds/), [`il_gen.py`](shivyc/il_gen.py), and `make_code` functions in [`tree/*.py`](shivyc/tree/)).
*   **Assembly (ASM) Generation:** Converts the IL to x86-64 assembly (implemented in [`asm_gen.py`](shivyc/asm_gen.py), using George and Appel's iterated register coalescing algorithm and  `make_asm` functions in [`il_cmds/*.py`](shivyc/il_cmds/)).

## Contributing

This project is no longer under active development, but contributions or issues are welcome.

*   **Questions:** Use GitHub Issues.
*   **Suggestions:** Submit an Issue with ideas for practical improvements.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf