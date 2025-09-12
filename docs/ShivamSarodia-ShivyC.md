# ShivyC: A C Compiler Written in Python 

**ShivyC is a hobby C compiler built in Python, offering a learning experience in compiler design.** [Explore the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC implements a subset of the C11 standard, enabling you to compile and execute C code.
*   **Optimized Binaries:** Generates reasonably efficient x86-64 binaries, including basic optimization techniques.
*   **Detailed Error Messages:** Provides helpful compile-time error messages, aiding in debugging.
*   **Educational Project:** A great resource for understanding the inner workings of a compiler.
*   **Written in Python:**  Leverages Python's readability for easy understanding and modification.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Compile and Run a Simple C Program

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

### Run Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker Environment

For non-Linux users or for a controlled environment, use Docker:

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker shell:

    ```bash
    docker/shell
    ```

    This opens a shell with ShivyC installed.

3.  Use ShivyC inside the Docker container:

    ```bash
    shivyc any_c_file.c           # To compile a file
    python3 -m unittest discover  # To run tests
    ```
    Changes in the local directory are reflected in the Docker environment.

## Implementation Overview

ShivyC's architecture includes these key components:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the input source code (primarily in `lexer.py`, with tokens defined in `tokens.py` and `token_kinds.py`).
*   **Parser:** Employs recursive descent parsing to create an abstract syntax tree (AST) in `parser/*.py` and `tree/*.py`.
*   **IL Generation:** Converts the AST into a custom intermediate language (IL) (commands in `il_cmds/*.py`, IL generation in `il_gen.py` and `tree/*.py`).
*   **ASM Generation:** Transforms the IL into x86-64 assembly code (implemented in `asm_gen.py` and `il_cmds/*.py`), with register allocation using George and Appel's iterated register coalescing.

## Contributing

While active development is limited, feedback and suggestions are welcome:

*   **Questions:**  Submit questions via GitHub Issues.
*   **Feature Requests:**  Propose practical improvements via GitHub Issues.

## References

*   [Original Compiler - ShivC](https://github.com/ShivamSarodia/ShivC)
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)