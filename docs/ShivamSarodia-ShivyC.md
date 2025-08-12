# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built entirely in Python that offers a unique approach to understanding compiler design.** [See the original repository](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

**Key Features:**

*   **C11 Standard Support:** ShivyC supports a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries with some optimizations.
*   **Helpful Error Messages:** Provides clear and informative compile-time error messages.
*   **Written in Python:** Leverages Python 3 for a clean and accessible implementation.
*   **Educational Project:** Offers a great learning resource for compiler design principles.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically already installed on Linux systems)

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Example

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

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker for Other Architectures

A Dockerfile is provided for an x86-64 Linux Ubuntu environment.

1.  Clone the repository and run:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    docker/shell
    ```

2.  Compile and test within the Docker container:

    ```bash
    shivyc any_c_file.c  # Compile a C file
    python3 -m unittest discover  # Run the tests
    ```

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the input code (implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:** Uses recursive descent parsing to create a parse tree (implemented in `parser/*.py` and `tree/*.py`).
*   **Intermediate Language (IL) Generation:** Converts the parse tree into a custom IL (implemented in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **Assembly (ASM) Generation:** Transforms IL commands into Intel-format x86-64 assembly code, including register allocation using George and Appel's iterated register coalescing algorithm (implemented in `asm_gen.py` and `il_cmds/*.py`).

## Contributing

This project is no longer under active development.  However, feel free to raise issues with questions and suggestions on how the project can be made practically helpful.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler, ShivyC.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf