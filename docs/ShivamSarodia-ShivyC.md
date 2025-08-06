# ShivyC: A C Compiler Written in Python

**ShivyC** is a hobby C compiler written in Python that allows you to compile a subset of the C11 standard, generating efficient binaries and providing helpful compile-time error messages.  [See the source code on GitHub](https://github.com/ShivamSarodia/ShivyC).

## Key Features

*   **C11 Subset Support:** Compiles a portion of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages.
*   **Built-in Optimizations:** Includes some optimizations for improved performance.
*   **Written in Python:** Leverages Python 3 for the compiler's implementation.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux systems)

### Installation
```bash
pip3 install shivyc
```

### Example Usage

Create a `hello.c` file:

```c
$ vim hello.c
$ cat hello.c

#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```

Compile and run:

```bash
shivyc hello.c
./out
```

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Other Architectures (Docker)

For non-Linux users, use the provided Dockerfile:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

This will open a shell with ShivyC installed.  You can then compile and test as described above within the Docker environment. Changes made to the local ShivyC directory are reflected in the Docker environment.

## Implementation Overview

ShivyC's compilation process includes the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (lexer.py, preproc.py).
*   **Lexer:** Converts source code into tokens (lexer.py, tokens.py, token_kinds.py).
*   **Parser:** Uses recursive descent to create a parse tree (parser/\*.py, tree/\*.py).
*   **IL Generation:** Creates a custom intermediate language (il\_cmds/\*.py, il\_gen.py, tree/\*.py).
*   **ASM Generation:** Translates IL into x86-64 assembly code (asm\_gen.py, il\_cmds/\*.py).  Register allocation uses George and Appel’s iterated register coalescing algorithm.

## Contributing

This project is no longer under active development.  However:

*   **Questions:** Use Github Issues to ask questions.
*   **Suggestions:**  Open an Issue with any perspective on how ShivyC can be made practically helpful.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - Original C compiler written by the author.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)