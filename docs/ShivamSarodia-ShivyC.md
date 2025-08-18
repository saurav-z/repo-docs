# ShivyC: A C Compiler Written in Python 🐍

**ShivyC is a hobby C compiler, written in Python, that lets you compile C code and experience how a compiler works.** Check out the original repository on GitHub: [ShivamSarodia/ShivyC](https://github.com/ShivamSarodia/ShivyC).

## Key Features

*   **C11 Standard Support:** ShivyC supports a subset of the C11 standard.
*   **Python-Based:** Built entirely in Python 3, making it easy to understand and modify.
*   **Generates Efficient Binaries:** Produces reasonably efficient x86-64 binaries, including some optimizations.
*   **Helpful Error Messages:** Provides clear compile-time error messages to assist with debugging.
*   **Intermediate Language (IL) Generation:** Uses a custom IL to simplify the compilation process.
*   **Assembly Generation:** Converts the IL into Intel-format x86-64 assembly code, including register allocation using the George and Appel's iterated register coalescing algorithm.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux systems)

### Installation

```bash
pip3 install shivyc
```

### Example: "Hello, World!"

1.  Create a file named `hello.c`:

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

### Docker Environment

For convenience, a Dockerfile is provided to set up an x86-64 Linux Ubuntu environment:

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker shell:

    ```bash
    docker/shell
    ```

    Within the Docker shell, you can compile and run tests:

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

## Implementation Overview

ShivyC's compilation process is broken down into the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Converts the source code into tokens (implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:** Uses recursive descent techniques to create a parse tree (implemented in `parser/*.py` and `tree/*.py`).
*   **IL Generation:** Transforms the parse tree into a custom intermediate language (IL), implemented in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`.
*   **ASM Generation:** Converts the IL into x86-64 assembly code, using register allocation (implemented in `asm_gen.py`, `il_cmds/*.py`, and `tree/*.py`).

## Contributing

Please note that the project is no longer under active development. However, if you have questions or suggestions, please create an issue.

## References

*   [ShivC (Older Compiler)](https://github.com/ShivamSarodia/ShivC) - The original C compiler that ShivyC is based on.
*   [C11 Specification](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   [x86_64 ABI](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   [Iterated Register Coalescing (George and Appel)](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)