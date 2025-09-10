# ShivyC: A C Compiler Written in Python

**ShivyC** is a hobby C compiler written in Python 3 that empowers you to compile a subset of the C11 standard and generate efficient binaries.  You can explore the project on GitHub: [https://github.com/ShivamSarodia/ShivyC](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:** Compiles a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Optimizations:** Includes optimizations to improve code performance.
*   **Compile-Time Error Messages:** Provides helpful error messages for easier debugging.
*   **Written in Python:** Built entirely in Python 3, making it accessible and easy to modify.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example: Compile and Run

1.  **Create a `hello.c` file:**

```c
$ vim hello.c
$ cat hello.c

#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```

2.  **Compile and Run:**

```bash
shivyc hello.c
./out
hello, world!
```

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker for Other Architectures

For users not running Linux, a Dockerfile is provided in the `docker/` directory to set up an x86-64 Linux Ubuntu environment.

1.  **Clone the repository:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  **Run the Docker shell:**

```bash
docker/shell
```

This will open a shell with ShivyC installed, allowing you to compile and test C code within the container.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (lexer.py and preproc.py).
*   **Lexer:** Tokenizes the input code (lexer.py, tokens.py, and token\_kinds.py).
*   **Parser:** Uses recursive descent to create a parse tree (parser/\*.py, tree/\*.py).
*   **IL Generation:** Converts the parse tree to a custom Intermediate Language (IL) (il\_cmds/\*.py, il\_gen.py, tree/\*.py).
*   **ASM Generation:** Translates IL commands to x86-64 assembly (asm\_gen.py, il\_cmds/\*.py). Includes register allocation using the George and Appel iterated register coalescing algorithm.

## Contributing

This project is not actively maintained, but contributions and feedback are still welcome.

*   Report any questions through GitHub Issues.
*   Share your perspectives on how ShivyC can be made practically helpful to a group by creating an Issue.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler, which ShivyC is a rewrite of.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf