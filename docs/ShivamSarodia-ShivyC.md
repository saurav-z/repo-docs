# ShivyC: A C Compiler Built in Python

**ShivyC is a hobby C compiler, written in Python, that brings you closer to understanding how compilers work.**  Check out the original repository for more information: [https://github.com/ShivamSarodia/ShivyC](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC supports a subset of the C11 standard and generates reasonably efficient x86-64 binaries, complete with optimizations, and helpful compile-time error messages.

**Key Features:**

*   **Written in Python:** Easily accessible and modifiable for those learning compiler design.
*   **C11 Subset Support:** Allows you to compile a subset of the modern C standard.
*   **Optimized Binaries:** Generates efficient x86-64 assembly code.
*   **Helpful Error Messages:** Provides clear and informative compile-time errors.
*   **Includes Trie Example:** See [this implementation of a trie](tests/general_tests/trie/trie.c) as an example of code ShivyC compiles.

## Quickstart

### x86-64 Linux

ShivyC requires only Python 3.6 or later to compile C code. You will need the GNU binutils and glibc, which you likely already have installed.

To install ShivyC:

```bash
pip3 install shivyc
```

To compile and run a "Hello, World!" program:

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

To run the tests:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Other Architectures (Using Docker)

For those not running Linux, use the provided Dockerfile to set up an x86-64 Linux Ubuntu environment with everything necessary for ShivyC:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

This opens a shell with ShivyC installed. Compile and run with:

```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

The Docker ShivyC executable updates live with changes in your local ShivyC directory.

## Implementation Overview

ShivyC's compilation process is broken down into the following stages:

*   **Preprocessor:**  Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Transforms source code into tokens (implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:**  Uses recursive descent to create a parse tree (implemented in `parser/*.py` and `tree/*.py`).
*   **IL Generation:**  Generates a custom intermediate language (IL) from the parse tree (implemented in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code, including register allocation using George and Appel’s iterated register coalescing algorithm (implemented in `asm_gen.py`, `il_cmds/*.py`).

## Contributing

This project is no longer under active development.

*   **Questions:** Open an issue on GitHub.
*   **Suggestions:**  Propose ideas for practical improvements via Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler that ShivyC is based on.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)