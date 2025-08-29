# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, written entirely in Python, that aims to provide a functional subset of the C11 standard.** Check out the original repository [here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:** Supports a subset of the C11 standard.
*   **Python-Based:** Written in Python 3 for ease of understanding and modification.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** Provides compile-time error messages to aid in debugging.
*   **Optimizations:** Includes some optimizations to improve generated code.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux systems)

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

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Using Docker (for other architectures or environments)

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker setup:

    ```bash
    docker/shell
    ```

    This will open a shell with ShivyC installed. Compile C files and run tests as described above. Changes to the local ShivyC directory are live-updated.

## Implementation Overview

A brief overview of the compiler's structure:

*   **Preprocessor:** Handles comments and `#include` directives (in `lexer.py` and `preproc.py`).
*   **Lexer:** Transforms the source code into tokens (in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:** Employs recursive descent to create a parse tree (in `parser/*.py` and `tree/*.py`).
*   **IL Generation:** Converts the parse tree into a custom Intermediate Language (IL) (in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Translates IL commands into x86-64 assembly code using register allocation (in `asm_gen.py` and `il_cmds/*.py`).

## Contributing

This project is not actively maintained, but the author welcomes questions and suggestions.

*   **Questions:** Post questions via Github Issues.
*   **Suggestions:** Make an Issue with perspectives on making ShivyC more helpful to a group.

## References

*   [ShivyC GitHub](https://github.com/ShivamSarodia/ShivC) - The author's previous C compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf