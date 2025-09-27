# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built in Python, designed to support a subset of the C11 standard and generate reasonably efficient x86-64 binaries.**  Explore the source code at the [ShivyC GitHub repository](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset:** Implements a portion of the C11 standard.
*   **Python-Based:**  Written entirely in Python 3, making it accessible and easy to understand.
*   **x86-64 Assembly Generation:** Produces x86-64 assembly code.
*   **Optimizations:** Includes some basic optimizations for improved performance.
*   **Compile-Time Error Messages:**  Provides helpful error messages to assist with debugging.
*   **Trie Implementation Example:** Includes an example implementation of a trie to demonstrate capabilities.

## Quickstart

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc

### Installation
```bash
pip3 install shivyc
```

### Example Usage
```c
// Create a hello.c file:
$ vim hello.c
$ cat hello.c
#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```
```bash
# Compile and Run:
$ shivyc hello.c
$ ./out
hello, world!
```

### Running Tests
```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker for Other Architectures
For those not running Linux, use the provided Dockerfile to set up an x86-64 Linux Ubuntu environment with ShivyC installed.

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

Then, within the Docker shell:
```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```
## Implementation Overview

ShivyC's compilation process is broken down into the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (lexer.py, preproc.py).
*   **Lexer:** Converts source code into tokens (lexer.py, tokens.py, token_kinds.py).
*   **Parser:** Uses recursive descent to build a parse tree (parser/\*.py, tree/\*.py).
*   **IL Generation:** Creates an intermediate language (IL) representation from the parse tree (il_cmds/\*.py, il_gen.py, tree/\*.py).
*   **ASM Generation:** Translates the IL into x86-64 assembly code, including register allocation using the George and Appel's iterated register coalescing algorithm (asm_gen.py, il_cmds/\*.py).

## Contributing

This project is no longer under active development. However:

*   For questions, please open a Github Issue.
*   For suggestions, please open a Github Issue.

## References

*   [ShivyC GitHub Repository](https://github.com/ShivamSarodia/ShivyC) - The original project repository.
*   [ShivC](https://github.com/ShivamSarodia/ShivC) - ShivyC is a rewrite from scratch of my old C compiler, ShivC, with much more emphasis on feature completeness and code quality. See the ShivC README for more details.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf