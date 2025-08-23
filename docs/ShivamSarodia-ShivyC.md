# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built in Python 3 that brings you closer to understanding how compilers work.**  Dive into the world of compiler design with ShivyC, a project aiming to support a subset of the C11 standard and generate efficient binaries. [Explore the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC).

## Key Features

*   **C11 Subset Support:** Implements a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries, with some optimizations.
*   **Helpful Error Messages:** Provides informative compile-time error messages for easier debugging.
*   **Written in Python:**  Built entirely in Python 3, making it accessible and easy to understand.

## Getting Started

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation
```bash
pip3 install shivyc
```

### Example Usage

1.  Create a `hello.c` file with the following content:
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

### Testing

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

For non-Linux users, use the provided Dockerfile:

1.  Clone the repository (if you haven't already)
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Start the Docker shell:
    ```bash
    docker/shell
    ```

    Inside the Docker shell, you can compile files with `shivyc` and run tests.

## Implementation Overview

ShivyC is broken down into the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in [`lexer.py`](shivyc/lexer.py) and [`preproc.py`](shivyc/lexer.py)).
*   **Lexer:** Transforms the source code into tokens (implemented in [`lexer.py`](shivyc/lexer.py), [`tokens.py`](shivyc/tokens.py), and [`token_kinds.py`](shivyc/token_kinds.py)).
*   **Parser:** Uses recursive descent to create a parse tree (located in [`parser/*.py`](shivyc/parser/) and [`tree/*.py`](shivyc/tree/)).
*   **Intermediate Language (IL) Generation:** Transforms the parse tree into a custom IL (defined in [`il_cmds/*.py`](shivyc/il_cmds/), [`il_gen.py`](shivyc/il_gen.py), and `make_code` functions in [`tree/*.py`](shivyc/tree/)).
*   **Assembly (ASM) Generation:** Converts the IL into x86-64 assembly code using iterated register coalescing (implemented in [`asm_gen.py`](shivyc/asm_gen.py) and `make_asm` functions in [`il_cmds/*.py`](shivyc/il_cmds/)).

## Contributing

This project is no longer under active development.  If you have questions, please submit an issue on GitHub.  Feedback on how ShivyC can be made helpful is appreciated.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The previous version of the C compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf