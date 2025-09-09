# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler written in Python that brings a subset of the C11 standard to life.**  ([View the original repo](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC compiles a subset of C11 and generates binaries with optimizations, all while providing helpful compile-time error messages.

**Key Features:**

*   **Written in Python 3:**  Leverages the versatility and readability of Python.
*   **C11 Subset Support:**  Implements a portion of the C11 standard.
*   **Optimized Binaries:** Produces reasonably efficient x86-64 binaries.
*   **Clear Error Messages:** Provides helpful messages to aid in debugging.
*   **Includes a preprocessor:** Parses comments and expands `#include` directives.
*   **Uses recursive descent techniques for all parsing** and generates a parse tree.
*   **Implements George and Appel’s iterated register coalescing algorithm** for register allocation.

## Getting Started

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example

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

The [`docker/`](docker/) directory provides a Dockerfile for easy setup on different systems.

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run Docker:

    ```bash
    docker/shell
    ```

    Now, you can compile and run from within the Docker environment:

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in [`lexer.py`](shivyc/lexer.py) and [`preproc.py`](shivyc/lexer.py)).
*   **Lexer:** Tokenizes the input (implemented in [`lexer.py`](shivyc/lexer.py), [`tokens.py`](shivyc/tokens.py), and [`token_kinds.py`](shivyc/token_kinds.py)).
*   **Parser:** Uses recursive descent to create a parse tree (implemented in [`parser/*.py`](shivyc/parser/) and [`tree/*.py`](shivyc/tree/)).
*   **IL Generation:** Generates a custom intermediate language (implemented in [`il_cmds/*.py`](shivyc/il_cmds/), [`il_gen.py`](shivyc/il_gen.py), and the `make_code` function in [`tree/*.py`](shivyc/tree/)).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code (implemented in [`asm_gen.py`](shivyc/asm_gen.py) and the `make_asm` function in [`il_cmds/*.py`](shivyc/il_cmds/)).

## Contributing

*   For questions, use Github Issues.
*   Suggestions for practical improvements are welcome via Issues.

## References

*   [ShivyC](https://github.com/ShivamSarodia/ShivC) - The original C compiler rewritten.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf