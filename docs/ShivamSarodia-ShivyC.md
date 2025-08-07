# ShivyC: A C Compiler in Python

**ShivyC is a hobby C compiler written in Python that supports a subset of the C11 standard and generates reasonably efficient binaries.**  You can find the original repository [here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC compiles a subset of the C11 standard.
*   **Python-Based:**  Built entirely in Python 3, making it accessible and easy to understand.
*   **x86-64 Assembly Generation:** Generates x86-64 assembly code.
*   **Optimization:** Includes some optimizations for improved binary performance.
*   **Helpful Error Messages:** Provides informative compile-time error messages to aid debugging.
*   **Uses George and Appel’s iterated register coalescing algorithm** Performs register allocation.

## Quickstart

### x86-64 Linux

ShivyC requires Python 3.6 or later, and utilizes GNU binutils and glibc for assembling and linking.

**Install:**
```bash
pip3 install shivyc
```
**Compile and Run a Hello World Program:**
```c
$ vim hello.c
$ cat hello.c

#include <stdio.h>
int main() {
  printf("hello, world!\n");
}

$ shivyc hello.c
$ ./out
hello, world!
```

**Run Tests:**
```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Other Architectures (Using Docker)

For users not on Linux, use the provided Dockerfile to create an Ubuntu x86-64 environment.

**Setup:**
```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

**Usage Inside Docker:**
```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

Changes made in your local ShivyC directory will update live in the Docker environment.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives. (lexer.py, preproc.py)
*   **Lexer:** Converts source code into tokens. (lexer.py, tokens.py, token_kinds.py)
*   **Parser:**  Uses recursive descent to build a parse tree. (parser/\*.py, tree/\*.py)
*   **IL Generation:**  Creates a custom intermediate language. (il\_cmds/\*.py, il\_gen.py, tree/\*.py)
*   **ASM Generation:** Converts IL commands into x86-64 assembly. (asm\_gen.py, il\_cmds/\*.py)

## Contributing

This project is no longer under active development. However:

*   Ask questions via Github Issues.
*   Suggest improvements or ideas via Issues.

## References

*   [ShivyC](https://github.com/ShivamSarodia/ShivC) - The original C compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf