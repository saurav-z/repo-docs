# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler written in Python that brings a subset of the C11 standard to life.** ([Original Repo](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC is a project focused on creating a C compiler using Python. It aims to support a subset of the C11 standard and generates reasonably efficient binaries with some optimizations. It also provides helpful compile-time error messages.

## Key Features

*   **Subset of C11 Standard:** Implements a portion of the C11 standard.
*   **Optimized Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Compile-Time Error Messages:** Provides helpful error messages for debugging.
*   **Written in Python:** Built entirely in Python 3.
*   **Includes a Preprocessor:** Parses comments and expands `#include` directives.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Example

Create a `hello.c` file:

```c
$ vim hello.c
$ cat hello.c

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

### Testing

Clone the repository and run the tests:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Using Docker (for non-Linux Environments)

The [`docker/`](docker/) directory provides a Dockerfile for an x86-64 Linux Ubuntu environment with ShivyC pre-installed.

1.  Clone the repository.
2.  Navigate to the project directory: `cd ShivyC`.
3.  Run: `docker/shell`

This will launch a shell where you can use `shivyc` to compile C files and run tests with `python3 -m unittest discover`. Changes made to your local ShivyC directory are reflected live in the Docker environment.

## Implementation Overview

ShivyC's architecture includes the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (`lexer.py`, `preproc.py`).
*   **Lexer:** Breaks down the source code into tokens (`lexer.py`, `tokens.py`, `token_kinds.py`).
*   **Parser:** Uses recursive descent techniques to build a parse tree (`parser/*.py`, `tree/*.py`).
*   **IL Generation:** Creates a custom intermediate language (IL) representation (`il_cmds/*.py`, `il_gen.py`, `tree/*.py`).
*   **ASM Generation:** Translates the IL into x86-64 assembly code, including register allocation using George and Appel’s iterated register coalescing algorithm (`asm_gen.py`, `il_cmds/*.py`).

## Contributing

*   **Questions:** Use Github Issues for questions.
*   **Suggestions:**  Suggest ideas via Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC): Original C compiler project.
*   C11 Specification: [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI: [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel): [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)