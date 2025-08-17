# ShivyC: A C Compiler in Python

**ShivyC is a hobby C compiler written in Python, offering a glimpse into compiler design and the inner workings of C programming.**

[View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC)

ShivyC is a project built to explore the intricacies of C compilation. This compiler supports a subset of the C11 standard and generates reasonably efficient x86-64 binaries, including some optimizations. It also provides helpful compile-time error messages.

## Key Features

*   **Written in Python 3:** Easy to understand and contribute to.
*   **C11 Standard Support:** Implements a subset of the C11 standard.
*   **x86-64 Binary Generation:** Produces executable binaries for Linux.
*   **Optimizations:** Includes optimizations for improved performance.
*   **Clear Error Messages:** Provides helpful error messages for easier debugging.

## Quickstart

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation
```bash
pip3 install shivyc
```

### Compile and Run a Simple Program

1.  Create a `hello.c` file with the following content:
    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```
2.  Compile and run the program:
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
2.  Run the tests:
    ```bash
    python3 -m unittest discover
    ```

### Using Docker (For Non-Linux Environments)
The [`docker/`](docker/) directory provides a Dockerfile that sets up an x86-64 Linux Ubuntu environment.

1.  Clone the repository:
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```
2.  Run the Docker shell:
    ```bash
    docker/shell
    ```
    This will open a shell with ShivyC installed.  Within the shell:
    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives. ([`lexer.py`](shivyc/lexer.py), [`preproc.py`](shivyc/preproc.py))
*   **Lexer:** Converts source code into tokens. ([`lexer.py`](shivyc/lexer.py), [`tokens.py`](shivyc/tokens.py), [`token_kinds.py`](shivyc/token_kinds.py))
*   **Parser:** Uses recursive descent to create a parse tree. ([`parser/*.py`](shivyc/parser/), [`tree/*.py`](shivyc/tree/))
*   **IL Generation:** Transforms the parse tree into a custom intermediate language. ([`il_cmds/*.py`](shivyc/il_cmds/), [`il_gen.py`](shivyc/il_gen.py),  [`tree/*.py`](shivyc/tree/))
*   **ASM Generation:** Converts IL commands into x86-64 assembly code. ([`asm_gen.py`](shivyc/asm_gen.py), [`il_cmds/*.py`](shivyc/il_cmds/))

## Contributing

This project is no longer under active development.  However, if you have questions or suggestions, please open a Github Issue.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler this project is based on.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf