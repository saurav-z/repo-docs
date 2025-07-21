# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built using Python that supports a subset of the C11 standard and generates efficient binaries.** You can find the original repository [here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC implements a portion of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries with some optimizations.
*   **Helpful Error Messages:** Provides useful compile-time error messages for easier debugging.
*   **Written in Python:** Built entirely in Python 3 for easy accessibility and modification.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux systems)

### Installation

```bash
pip3 install shivyc
```

### Example Usage

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
shivyc hello.c
./out
# Output: hello, world!
```

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker Quickstart

For users not running Linux, a Dockerfile is provided for a convenient development environment:

1.  Clone the repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

This will launch a shell within a Docker container with ShivyC installed. You can then compile and test your code directly within the container:

```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives. Implemented in `lexer.py` and `preproc.py`.
*   **Lexer:** Converts source code into tokens. Implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`.
*   **Parser:** Uses recursive descent to create a parse tree.  Implemented in `parser/*.py` and creates nodes defined in `tree/*.py`.
*   **IL Generation:** Generates a custom intermediate language (IL) by traversing the parse tree.  Implemented primarily in `tree/*.py` and `il_gen.py`.
*   **ASM Generation:** Converts IL commands into x86-64 assembly code.  Includes register allocation using the George and Appel algorithm. Implemented in `asm_gen.py` and `il_cmds/*.py`.

## Contributing

This project is no longer under active development. For questions, use GitHub Issues.  For suggestions on making ShivyC more useful, create an Issue.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler, from which ShivyC was rewritten.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf