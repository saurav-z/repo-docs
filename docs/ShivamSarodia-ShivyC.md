# ShivyC: A Hobby C Compiler Written in Python

**ShivyC** is a C compiler built from the ground up in Python, providing a hands-on learning experience for compiler design and a functional C compiler. [Check out the original repository](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC supports a subset of the C11 standard, allowing you to compile a variety of C code.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries with some optimizations.
*   **Helpful Error Messages:** Provides informative compile-time error messages to aid in debugging.
*   **Written in Python:** Designed as a learning tool, the codebase is Python-based for easier understanding and modification.
*   **Complete Compilation Process:** Includes preprocessor, lexer, parser, intermediate language (IL) generation, and assembly code generation.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

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

    This will compile the C code and execute the resulting binary, printing "hello, world!".

### Testing

To run the tests, clone the repository and use the unittest module:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker Quickstart (for non-Linux users)

The `docker/` directory provides a Dockerfile to set up an x86-64 Linux Ubuntu environment with everything you need.

1.  Clone the repository.

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Build and use the Docker container:

    ```bash
    docker/shell
    ```

    Inside the container, you can then compile and run your C files with `shivyc any_c_file.c` and run the tests with `python3 -m unittest discover`.

## Implementation Overview

ShivyC's compilation process is broken down into these key components:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Converts source code into tokens (implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:** Uses recursive descent parsing to build a parse tree (implemented in `parser/*.py` and `tree/*.py`).
*   **IL Generation:** Generates a custom intermediate language from the parse tree (`il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code, including register allocation using the George and Appel's iterated register coalescing algorithm (`asm_gen.py`, `il_cmds/*.py`).

## Contributing

While this project is no longer under active development, contributions and questions are welcome.

*   **Questions:** Use Github Issues to ask questions.
*   **Suggestions:**  Suggest ideas via Issues to improve the project.

## References

*   [ShivyC GitHub](https://github.com/ShivamSarodia/ShivC) - The original C compiler that ShivyC is based on.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)