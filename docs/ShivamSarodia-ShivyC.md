# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler implemented in Python that brings you closer to understanding how compilers work.**  [View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset:** Supports a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries with some optimizations.
*   **Helpful Error Messages:** Provides informative compile-time error messages to aid in debugging.
*   **Written in Python:**  Easy to understand and modify, perfect for learning about compiler design.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example: Compile and Run

1.  **Create a C file (e.g., `hello.c`):**

```c
#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```

2.  **Compile:**

```bash
shivyc hello.c
```

3.  **Run:**

```bash
./out
```

This will output: `hello, world!`

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

## Docker Support

For users not on Linux, the project includes a Dockerfile for easy setup:

1.  **Clone the repository:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  **Enter the Docker shell:**

```bash
docker/shell
```

Inside the Docker environment, you can compile and test:

```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

Changes made in your local ShivyC directory are reflected in the Docker environment.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:**  Tokenizes the input code (primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:**  Uses recursive descent parsing to create a parse tree (in `parser/*.py`, using tree nodes defined in `tree/*.py`).
*   **Intermediate Language (IL) Generation:** Converts the parse tree to a custom IL (in `il_cmds/*.py`, with objects in `il_gen.py` and code primarily in `tree/*.py`).
*   **Assembly (ASM) Generation:** Transforms the IL into x86-64 assembly code (in `asm_gen.py`, with code primarily in `il_cmds/*.py`) using George and Appel’s iterated register coalescing algorithm for register allocation.

## Contributing

The project is no longer under active development but is open source. If you have a question, please open a Github Issue.

## References

*   [ShivC (Older Compiler)](https://github.com/ShivamSarodia/ShivC)
*   C11 Specification: http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI: https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel): https://www.cs.purdue.edu/homes/hosking/502/george.pdf