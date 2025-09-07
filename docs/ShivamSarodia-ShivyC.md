# ShivyC: A C Compiler Written in Python

**ShivyC** is a hobbyist C compiler, offering a fascinating look into the world of compiler design.  [View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:**  Implements a subset of the C11 standard.
*   **Python-Based:** Built entirely in Python 3.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Compile-Time Error Messages:** Provides helpful error messages to aid in debugging.
*   **Includes Optimizations:** Leverages optimizations to improve the performance of generated code.

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

Compile and Run:

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

### Docker Environment

For users not running Linux, a Docker environment is available:

1.  Clone the repository:
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```
2.  Build and enter the Docker shell:
    ```bash
    docker/shell
    ```
    Now you can compile and run C code within the Docker container.  Changes made in the host directory will be reflected.

## Implementation Overview

A high-level overview of the key components:

*   **Preprocessor:** Handles comments and `#include` directives (lexer.py, preproc.py).
*   **Lexer:** Breaks down the code into tokens (lexer.py, tokens.py, token\_kinds.py).
*   **Parser:** Uses recursive descent to create a parse tree (parser/\*.py, tree/\*.py).
*   **IL Generation:** Transforms the parse tree into a custom intermediate language (il\_cmds/\*.py, il\_gen.py, tree/\*.py).
*   **ASM Generation:** Converts the IL into x86-64 assembly code with register allocation (asm\_gen.py, il\_cmds/\*.py).

## Contributing

While the project is no longer under active development, contributions are welcome through GitHub Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The previous C compiler project.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)