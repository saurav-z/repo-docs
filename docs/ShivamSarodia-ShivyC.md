# ShivyC: A C Compiler Built in Python

**ShivyC is a hobby C compiler written in Python that brings you closer to understanding the inner workings of compilation.** Check out the original repository [here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:** Implements a subset of the C11 standard.
*   **Optimized Binaries:** Generates reasonably efficient binaries with built-in optimizations.
*   **Helpful Error Messages:** Provides informative compile-time error messages.
*   **Written in Python:** Leverages the versatility and readability of Python for compiler development.

## Quickstart

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation
```bash
pip3 install shivyc
```

### Example: Compile and Run

1.  **Create a C file (e.g., `hello.c`):**
    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  **Compile:**
    ```bash
    shivyc hello.c
    ```

3.  **Run:**
    ```bash
    ./out
    ```
    This will print "hello, world!" to your console.

### Running Tests
```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Other Architectures (using Docker)

The [`docker/`](docker/) directory provides a Dockerfile for an x86-64 Linux Ubuntu environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  **Enter the Docker shell:**
    ```bash
    docker/shell
    ```
    This will open a shell where `shivyc` and the tests can be run.

## Implementation Overview

### 1. Preprocessor
Handles comments and `#include` directives. Implemented in `lexer.py` and `preproc.py`.

### 2. Lexer
Breaks down the source code into tokens.  Implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`.

### 3. Parser
Uses recursive descent to create a parse tree from the tokens. Implemented in the `parser/` directory, creating nodes defined in `tree/*.py`.

### 4. IL Generation
Traverses the parse tree to generate a custom intermediate language (IL). IL commands are defined in `il_cmds/*.py`, and most of the IL generation occurs in the `make_code` functions in `tree/*.py`.

### 5. ASM Generation
Converts IL commands into x86-64 assembly code. Implemented in `asm_gen.py`, and the majority of the ASM generation is in the `make_asm` functions of each IL command in `il_cmds/*.py`. Utilizes George and Appel's iterated register coalescing algorithm for register allocation.

## Contributing

While active development is limited, questions and suggestions are welcome.

*   **Questions:** Open a Github Issue.
*   **Suggestions:** Open a Github Issue with a specific perspective on how ShivyC can be made practically helpful to a group.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The previous version of this compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf