# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, built in Python, that allows you to compile and run C code.** ([View on GitHub](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC supports a subset of the C11 standard and generates reasonably efficient binaries with some optimizations, complete with helpful compile-time error messages.

## Key Features

*   **C11 Subset Support:** Compiles a portion of the C11 standard.
*   **Written in Python:** Leverages the versatility of Python for compiler development.
*   **x86-64 Binaries:** Generates binaries compatible with the x86-64 architecture.
*   **Optimizations:** Includes optimizations to improve the efficiency of compiled code.
*   **Error Messages:** Provides helpful compile-time error messages to aid debugging.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically already installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example

1.  Create a C file (e.g., `hello.c`):

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

1.  Clone the repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  Run tests:

```bash
python3 -m unittest discover
```

### Docker (for other architectures)

For users not running Linux, a Dockerfile is provided for ease of use.

1.  Clone the repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  Run the Docker shell:

```bash
docker/shell
```

This will provide an environment with ShivyC installed. Compile and run files as usual:
```bash
shivyc any_c_file.c
python3 -m unittest discover  # to run tests
```
The Docker ShivyC executable will update live with any changes made in your local ShivyC directory.

## Implementation Overview

ShivyC's compiler pipeline includes the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the input (primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Employs recursive descent parsing to build a parse tree (in `parser/*.py` and `tree/*.py`).
*   **IL Generation:** Converts the parse tree into a custom intermediate language (in `il_cmds/*.py`, `il_gen.py`, and the `make_code` functions in `tree/*.py`).
*   **ASM Generation:** Translates IL commands into x86-64 assembly code (in `asm_gen.py` and the `make_asm` functions in `il_cmds/*.py`). Register allocation uses George and Appel's iterated register coalescing algorithm.

## Contributing

While active development is limited, you can still contribute.

*   **Questions:** Ask questions via GitHub Issues.
*   **Suggestions:** Propose ideas for improvement through Issues.

## References

*   [ShivC (Previous Compiler)](https://github.com/ShivamSarodia/ShivC)
*   [C11 Specification](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   [x86_64 ABI](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   [Iterated Register Coalescing (George and Appel)](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)