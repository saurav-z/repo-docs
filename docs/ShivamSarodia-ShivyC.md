# ShivyC: A C Compiler in Python

**ShivyC is a hobby C compiler, written in Python, that lets you compile C code and explore compiler design.** Explore the inner workings of a compiler and learn how C code gets translated into executable binaries with ShivyC! View the original repository [here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC compiles a significant subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries with some optimizations.
*   **Helpful Error Messages:** Provides informative compile-time error messages to aid in debugging.
*   **Written in Python:** Leverage your Python skills to understand and contribute to a compiler.
*   **Intermediate Representation (IL):** Uses a custom intermediate language for code transformation.
*   **Register Allocation:** Implements register allocation using the George and Appel algorithm.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Example: Hello World

1.  Create a `hello.c` file:

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

To run the tests, clone the repository and execute:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker Environment

For users not on Linux, a Docker environment is available for easy setup:

1.  Clone the repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

2.  Compile and test within the Docker shell:

```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

## Implementation Overview

ShivyC is broken down into several key stages:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Converts source code into tokens (implemented primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Uses recursive descent to create a parse tree (in `parser/*.py` and `tree/*.py`).
*   **IL Generation:** Transforms the parse tree into a custom intermediate language (implemented in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Converts the IL into x86-64 assembly code, including register allocation using the George and Appel algorithm (in `asm_gen.py` and `il_cmds/*.py`).

## Contributing

The project is not actively maintained.  If you have questions, please use Github Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)