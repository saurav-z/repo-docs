# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built in Python 3, designed to compile a subset of the C11 standard and generate efficient binaries.** Check it out on [GitHub](https://github.com/ShivamSarodia/ShivyC)!

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **Subset of C11 Standard:** Supports a portion of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Optimizations:** Includes several optimizations for better performance.
*   **Helpful Error Messages:** Provides clear and informative compile-time error messages.
*   **Written in Python 3:** Leverages the power and readability of Python.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Compile and Run a C Program

1.  Create a `hello.c` file with the following content:

    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  Compile the program:

    ```bash
    shivyc hello.c
    ```

3.  Run the compiled executable:

    ```bash
    ./out
    ```

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker for Other Architectures

For users not on Linux, a Dockerfile is provided in the [`docker/`](docker/) directory to set up an x86-64 Ubuntu environment.

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    docker/shell
    ```

2.  Use the compiler within the Docker environment:

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

    *The Docker ShivyC executable will update live with any changes made in your local ShivyC directory.*

## Implementation Overview

*   **Preprocessor:** Handles comments and includes (`lexer.py`, `preproc.py`).
*   **Lexer:** Converts source code into tokens (`lexer.py`, `tokens.py`, `token_kinds.py`).
*   **Parser:** Uses recursive descent to build a parse tree (`parser/*.py`, `tree/*.py`).
*   **IL Generation:** Creates a custom intermediate language (`il_cmds/*.py`, `il_gen.py`, `tree/*.py`).
*   **ASM Generation:** Converts IL to x86-64 assembly code, using register allocation (George and Appel's algorithm) (`asm_gen.py`, `il_cmds/*.py`).

## Contributing

*   The project is no longer under active development.
*   Questions can be asked via Github Issues.
*   Suggestions for practical improvements are welcome as Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf