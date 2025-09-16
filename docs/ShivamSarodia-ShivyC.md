# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built in Python, providing a hands-on approach to understanding compiler design.**  You can find the original repository [here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Subset:** Supports a portion of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Compile-Time Error Messages:** Provides helpful error messages to aid in debugging.
*   **Written in Python:**  A great project to explore compiler internals in Python.
*   **Includes Optimizations:** ShivyC incorporates optimizations during compilation.

## Quickstart

### Requirements
*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example Usage

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

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker for Other Architectures

A Dockerfile is available in the `docker/` directory to provide an x86-64 Linux environment.

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

Inside the Docker shell:

```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives. (lexer.py & preproc.py)
*   **Lexer:** Transforms source code into tokens. (lexer.py, tokens.py, token_kinds.py)
*   **Parser:**  Employs recursive descent to build a parse tree. (parser/\*.py, tree/\*.py)
*   **IL Generation:** Creates a custom intermediate language. (il\_cmds/\*.py, il\_gen.py, tree/\*.py)
*   **ASM Generation:** Converts IL commands into x86-64 assembly, using George and Appel's register coalescing algorithm. (asm\_gen.py, il\_cmds/\*.py)

## Contributing

The project is not actively maintained. If you have questions, please use GitHub Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf