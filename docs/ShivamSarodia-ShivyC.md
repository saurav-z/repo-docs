# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, written in Python, that allows you to compile C code and generate efficient binaries.**  Explore the inner workings of a C compiler and gain a deeper understanding of programming language implementation!

[View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC)

**Key Features:**

*   **Subset of C11 Standard:** Supports a subset of the C11 standard for compiling C code.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Optimizations:** Includes some optimizations for improved performance.
*   **Compile-Time Error Messages:** Provides helpful error messages to aid in debugging.
*   **Written in Python:** Fully implemented in Python 3, making it accessible and easy to understand.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (for assembling and linking on Linux)

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Example: Compile and Run "Hello, World!"

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
    $ shivyc hello.c
    $ ./out
    hello, world!
    ```

### Running Tests

To run the tests, clone the repository and use `unittest`:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker Environment

For those not running Linux, use the provided Dockerfile:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

This will open a shell with ShivyC installed and ready to use. Compile with `shivyc any_c_file.c` and run tests with `python3 -m unittest discover`. Changes to the local ShivyC directory are reflected live in the Docker environment.

## Implementation Overview

ShivyC's compilation process consists of several key stages:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Breaks down the source code into tokens (implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:** Constructs a parse tree from the tokens using recursive descent (implemented in `parser/*.py` and `tree/*.py`).
*   **IL Generation:** Converts the parse tree into a custom Intermediate Language (IL) (implemented in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Translates the IL into x86-64 assembly code, including register allocation using George and Appel’s iterated register coalescing algorithm (implemented in `asm_gen.py`, and `il_cmds/*.py`).

## Contributing

This project is no longer under active development. If you have a question, please create a GitHub Issue.

## References

*   [ShivC (Original Compiler)](https://github.com/ShivamSarodia/ShivC)
*   [C11 Specification](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   [x86_64 ABI](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   [Iterated Register Coalescing (George and Appel)](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)