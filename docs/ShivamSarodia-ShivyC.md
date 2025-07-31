# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler that allows you to compile C code into reasonably efficient binaries, all written in Python.** Explore the source code and learn how compilers work! Check out the original project [here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:** ShivyC supports a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages.
*   **Written in Python:** The entire compiler is implemented in Python 3.
*   **Includes Optimizations:** The compiler incorporates optimizations to improve performance.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

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
```

## Other Architectures (Docker)

For users not on Linux, a Dockerfile is provided to set up an x86-64 Ubuntu environment with ShivyC:

1.  Clone the repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  Run the Docker shell:

```bash
docker/shell
```

This will provide a shell with ShivyC pre-installed. Use the following commands inside the Docker container:

```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

## Implementation Overview

ShivyC's compilation process is broken down into the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Transforms source code into tokens (implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:** Constructs a parse tree using recursive descent techniques (implemented in `parser/*.py` and `tree/*.py`).
*   **IL Generation:** Creates a custom intermediate language (IL) from the parse tree (implemented in `il_cmds/*.py`, `il_gen.py`, and the `make_code` functions in `tree/*.py`).
*   **ASM Generation:** Converts the IL into x86-64 assembly code, including register allocation using George and Appel's iterated register coalescing algorithm (implemented in `asm_gen.py`, `il_cmds/*.py`, and the `make_asm` functions).

## Contributing

The project is no longer under active development, and pull requests are unlikely to be reviewed.  However:

*   For questions, please create a Github Issue.
*   For suggestions on practical applications, please create a Github Issue.

## References

*   [ShivyC](https://github.com/ShivamSarodia/ShivC) - The original C compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf