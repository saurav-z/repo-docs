# ShivyC: A C Compiler in Python

**ShivyC is a hobby C compiler, written in Python, that allows you to compile C code into efficient binaries.** [(See Original Repo)](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC supports a subset of the C11 standard and generates x86-64 binaries, including optimizations, and helpful compile-time error messages.

**Key Features:**

*   **C11 Subset Support:** Compiles a significant portion of the C11 standard.
*   **Optimized Code Generation:** Produces reasonably efficient x86-64 binaries.
*   **Clear Error Messages:** Provides helpful compile-time error messages for debugging.
*   **Written in Python:** Leverages the versatility of Python for compiler development.

## Quickstart

### x86-64 Linux

ShivyC requires only Python 3.6 or later and utilizes the GNU binutils and glibc for assembling and linking, which are likely already installed on your system.

**Installation:**

```bash
pip3 install shivyc
```

**Example Usage:**

1.  Create a C file (e.g., `hello.c`):

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
hello, world!
```

**Running Tests:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Other Architectures (Using Docker)

For those not running Linux, the `docker/` directory provides a Dockerfile to set up an x86-64 Linux Ubuntu environment with everything needed for ShivyC.

1.  Clone the Repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  Enter the Docker Shell:

```bash
docker/shell
```

This opens a shell with ShivyC installed, allowing you to compile C files and run tests:

```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

Changes made in your local ShivyC directory are live-updated in the Docker environment.

## Implementation Overview

*   **Preprocessor:**  Parses comments and handles `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Transforms the input source code into tokens (`lexer.py`, `tokens.py`, `token_kinds.py`).
*   **Parser:** Utilizes recursive descent to build a parse tree (`parser/*.py`, `tree/*.py`).
*   **IL Generation:** Traverses the parse tree to generate a custom IL (intermediate language) (`il_cmds/*.py`, `il_gen.py`, `tree/*.py`).
*   **ASM Generation:** Converts IL commands into Intel-format x86-64 assembly code, including register allocation using George and Appel's iterated register coalescing algorithm (`asm_gen.py`, `il_cmds/*.py`).

## Contributing

This project is no longer under active development.

*   **Questions:** Use Github Issues for any inquiries.
*   **Suggestions:** Submit Issues with perspectives on how ShivyC can be made practically helpful to a group.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler from which ShivyC was rewritten.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf