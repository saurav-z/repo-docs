# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, crafted in Python, that brings a subset of the C11 standard to life with efficient binaries and helpful error messages.**  Explore the inner workings of a compiler with this educational project.  Check out the original repository for more details:  [https://github.com/ShivamSarodia/ShivyC](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:** Implements a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages.
*   **Written in Python:**  Built entirely using Python 3.
*   **Optimizations:** Includes some optimizations for better performance.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation

Install ShivyC using pip:

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

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the tests:

    ```bash
    python3 -m unittest discover
    ```

### Docker Environment

For those not using Linux, a Dockerfile is provided in the `docker/` directory to set up an x86-64 Ubuntu environment with ShivyC pre-installed.

1.  Clone the repository (if you haven't already):

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker setup:

    ```bash
    docker/shell
    ```

3.  Inside the Docker shell, compile and test:

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Breaks down the code into tokens (implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:** Uses recursive descent to create a parse tree (implemented in `parser/*.py` and `tree/*.py`).
*   **IL Generation:**  Generates a custom intermediate language (IL) (implemented in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code (implemented in `asm_gen.py`, `il_cmds/*.py`, and register allocation using George and Appel's iterated register coalescing algorithm).

## Contributing

*   This project is no longer under active development.
*   Questions can be asked by opening a Github Issue.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - Original C compiler
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf