# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler implemented in Python, offering a glimpse into the inner workings of a compiler and generating reasonably efficient binaries for a subset of the C11 standard.** ([Original Repository](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC) [![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC supports a subset of the C11 standard.
*   **Python-Based:** Written entirely in Python 3.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages.
*   **Includes Optimizations:** The compiler includes optimizations to improve generated code.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Example Usage

1.  Create a simple C program (e.g., `hello.c`):

```c
#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```

2.  Compile the program:

```bash
shivyc hello.c
```

3.  Run the compiled executable:

```bash
./out
```

### Running Tests

1.  Clone the repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  Run the tests:

```bash
python3 -m unittest discover
```

### Docker Support

For non-Linux users, the `docker/` directory provides a Dockerfile to set up an x86-64 Ubuntu environment with ShivyC.

1.  Clone the repository (if you haven't already):

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  Run the Docker shell:

```bash
docker/shell
```

This provides a shell with ShivyC pre-installed, allowing you to compile and test C files within the Docker environment.

## Implementation Overview

ShivyC's compilation process consists of the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Converts the source code into tokens (implemented primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Uses recursive descent to parse the tokens and create a parse tree (implemented in `parser/*.py` and `tree/*.py`).
*   **IL Generation:** Transforms the parse tree into a custom Intermediate Language (IL) (implemented in `il_cmds/*.py`, `il_gen.py`, and the `make_code` functions in `tree/*.py`).
*   **ASM Generation:** Converts the IL into x86-64 assembly code (implemented in `asm_gen.py`, and the `make_asm` functions in `il_cmds/*.py`).  Register allocation uses George and Appel’s iterated register coalescing algorithm.

## Contributing

This project is no longer under active development.  However, if you have questions or suggestions, please create an Issue on GitHub.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - ShivyC is a rewrite from scratch of my old C compiler, ShivC, with much more emphasis on feature completeness and code quality. See the ShivC README for more details.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf