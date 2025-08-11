# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, written entirely in Python, designed to compile a subset of the C11 standard.** ([Original Repository](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC offers a unique perspective on compiler design, implemented in Python. It supports a subset of the C11 standard and generates reasonably efficient x86-64 binaries with optimizations. Its design also includes helpful compile-time error messages.

## Key Features

*   **C11 Subset Support:** Compiles a portion of the C11 standard.
*   **Python-Based:** Written entirely in Python 3, making it accessible and easy to understand for Python developers.
*   **Optimized Binaries:** Generates x86-64 assembly with some built-in optimizations.
*   **Helpful Error Messages:** Provides informative compile-time error messages to aid in debugging.
*   **Open Source:** Available on GitHub for anyone to use.

## Quickstart

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc (usually already installed on Linux systems)

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Example: Hello World

Create a `hello.c` file:

```c
#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```

Compile and run:

```bash
shivyc hello.c
./out
```

### Running Tests

To run the tests, clone the repository and execute:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker Environment

For users not on Linux, a Dockerfile is provided in the [`docker/`](docker/) directory, which sets up an x86-64 Linux Ubuntu environment:

1.  Clone the repository.
2.  Navigate to the ShivyC directory.
3.  Run `docker/shell` to open a shell with ShivyC installed and ready.

    *   Compile a C file: `shivyc any_c_file.c`
    *   Run tests: `python3 -m unittest discover`

## Implementation Overview

ShivyC's compilation process is broken down into the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (`lexer.py`, `preproc.py`).
*   **Lexer:** Converts source code into tokens (`lexer.py`, `tokens.py`, `token_kinds.py`).
*   **Parser:** Uses recursive descent to create a parse tree (`parser/*.py`, `tree/*.py`).
*   **Intermediate Language (IL) Generation:** Transforms the parse tree into a custom IL (`il_cmds/*.py`, `il_gen.py`, `tree/*.py`).
*   **Assembly Generation:** Converts IL commands into x86-64 assembly code, including register allocation using the George and Appel iterated register coalescing algorithm (`asm_gen.py`, `il_cmds/*.py`).

## Contributing

While active development has ceased, you can still:

*   Ask questions via Github Issues.
*   Suggest ideas or improvements through Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler, which served as the foundation for ShivyC.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf