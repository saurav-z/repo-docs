# ShivyC: A C Compiler Written in Python

**Tired of complex compilers? ShivyC offers a simplified, Python-based approach to compiling a subset of the C11 standard.** Explore the project on [GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC compiles a significant subset of the C11 standard.
*   **Python-Based:** Written entirely in Python 3, making it accessible and easy to understand.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Compile-Time Error Messages:** Provides helpful error messages to aid debugging.
*   **Includes Optimizations:** ShivyC includes basic optimizations to improve performance.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example: Compile and Run

1.  Create a `hello.c` file:

    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  Compile using ShivyC:

    ```bash
    shivyc hello.c
    ```

3.  Run the compiled program:

    ```bash
    ./out
    ```

    This will print "hello, world!" to your console.

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

### Docker for Other Architectures

For users not running Linux, a Dockerfile is provided to set up an x86-64 Linux Ubuntu environment.

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker setup:

    ```bash
    docker/shell
    ```

    This will open a shell within a ShivyC-ready environment.  You can then compile and test as described above.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives, located in `lexer.py` and `preproc.py`.
*   **Lexer:** Converts source code into tokens, implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`.
*   **Parser:** Uses recursive descent parsing in the `parser/` directory, building a parse tree defined in `tree/`.
*   **IL Generation:** Translates the parse tree into a custom intermediate language (IL), with commands in `il_cmds/` and related code in `il_gen.py` and `tree/`.
*   **ASM Generation:** Converts IL commands into x86-64 assembly code, using the George and Appel iterated register coalescing algorithm.  Assembly generation is primarily in `asm_gen.py` and the `make_asm` functions of the IL commands.

## Contributing

This project is no longer under active development.  However, if you have questions, please raise an Issue. Contributions are unlikely to be accepted.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler this project is based on.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)