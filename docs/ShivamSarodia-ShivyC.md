# ShivyC: A C Compiler in Python

**ShivyC is a hobby C compiler written in Python 3, offering a unique approach to compiling C code.** [View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **Written in Python 3:** Leverages the versatility and readability of Python.
*   **C11 Subset Support:** Compiles a subset of the C11 standard.
*   **Optimized Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages.
*   **Includes a limited preprocessor:** Parses out comments and expands `#include` directives.

## Quickstart

### x86-64 Linux

**Prerequisites:** Python 3.6+ and GNU binutils/glibc (likely already installed).

**Installation:**

```bash
pip3 install shivyc
```

**Example Usage:**

1.  Create a `hello.c` file:

```c
$ vim hello.c
$ cat hello.c

#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```

2.  Compile and Run:

```bash
shivyc hello.c
./out
```

**Running Tests:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Other Architectures (Using Docker)

For users not running Linux, a Dockerfile is available for easy setup.

1.  Clone the repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  Run the Docker shell:

```bash
docker/shell
```

3.  Inside the Docker environment:

```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

## Implementation Overview

*   **Preprocessor:** Implemented between `lexer.py` and `preproc.py`. Handles comments and `#include` directives.
*   **Lexer:** Located in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`.
*   **Parser:** Utilizes recursive descent techniques, found in the `parser/` directory. Creates parse trees defined in `tree/`.
*   **IL Generation:** Traverses the parse tree to generate a custom IL (intermediate language). Commands for the IL are in `il_cmds/`. Most code is within the `make_code` function in the `tree/` directory.
*   **ASM Generation:** Converts IL commands into x86-64 assembly code. Register allocation uses George and Appel’s iterated register coalescing algorithm.  General functionality is in `asm_gen.py`. Most code is within the `make_asm` function in the `il_cmds/` directory.

## Contributing

This project is no longer under active development. If you have questions, please open a GitHub issue. If you have suggestions for improvements, please also create an issue.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The previous iteration of this compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf