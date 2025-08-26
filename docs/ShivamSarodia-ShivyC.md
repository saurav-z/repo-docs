# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby-grade C compiler written in Python that aims to support a subset of the C11 standard, providing efficient binaries and helpful error messages.** Check out the original repository [here](https://github.com/ShivamSarodia/ShivyC)!

## Key Features

*   **C11 Subset Support:** ShivyC currently supports a subset of the C11 standard, with ongoing development to expand compatibility.
*   **Efficient Binaries:** The compiler generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** ShivyC provides detailed compile-time error messages to aid in debugging.
*   **Built-in Optimizations:** Includes several optimizations to enhance code performance.
*   **Written in Python:** Leverages the versatility and readability of Python for compiler development.

## Quickstart

### x86-64 Linux

ShivyC is easy to get started with, requiring only Python 3.6 or later and the GNU binutils and glibc.

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

2.  Compile and run:

```bash
shivyc hello.c
./out
```

**Testing:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Other Architectures (using Docker)

For non-Linux users, a Dockerfile is provided to set up an x86-64 Ubuntu environment with everything needed:

1.  Clone the repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

2.  Compile and test inside the Docker container:

```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

## Implementation Overview

ShivyC's architecture includes the following key components:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the source code (implemented primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`).
*   **Parser:** Employs recursive descent parsing (implemented in `parser/*.py`) to create a parse tree (defined in `tree/*.py`).
*   **IL Generation:** Generates a custom intermediate language (IL) by traversing the parse tree (commands in `il_cmds/*.py`, objects in `il_gen.py`, and code within `tree/*.py`).
*   **ASM Generation:** Converts IL commands into Intel-format x86-64 assembly code. This includes register allocation using George and Appel's iterated register coalescing algorithm (`asm_gen.py` and `il_cmds/*.py`).

## Contributing

This project is no longer under active development, but feel free to use it and ask questions.

*   Ask questions about ShivyC via Github Issues.
*   Suggest ideas via an Issue.

## References

*   [ShivC (older compiler)](https://github.com/ShivamSarodia/ShivC) - The original C compiler that ShivyC is a rewrite of.
*   [C11 Specification](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   [x86_64 ABI](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf