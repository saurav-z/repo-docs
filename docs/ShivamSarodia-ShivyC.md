# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, written in Python, that aims to compile a subset of the C11 standard.** Check out the [original repository](https://github.com/ShivamSarodia/ShivyC) for more details.

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** Implements a portion of the C11 standard.
*   **Python 3:** Built entirely using Python 3.
*   **x86-64 Binaries:** Generates reasonably efficient binaries for the x86-64 architecture.
*   **Optimizations:** Includes various optimizations to improve performance.
*   **Compile-Time Errors:** Provides helpful error messages to aid in debugging.
*   **Intermediate Language (IL) and Assembly Generation:** Converts C code into a custom IL and then into x86-64 assembly.
*   **Register Allocation:** Implements register allocation using the George and Appel iterated register coalescing algorithm.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example Usage

1.  **Create a C file (e.g., `hello.c`):**

```c
#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```

2.  **Compile and Run:**

```bash
shivyc hello.c
./out
```

### Testing

1.  **Clone the repository:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  **Run tests:**

```bash
python3 -m unittest discover
```

### Docker for Other Architectures

For users not running Linux, a Dockerfile is provided to set up an Ubuntu environment with ShivyC.

1.  **Clone the repository:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  **Run the Docker shell:**

```bash
docker/shell
```

This will open a shell with ShivyC installed. You can then compile and test within the Docker environment:

```bash
shivyc any_c_file.c           # to compile a file
python3 -m unittest discover  # to run tests
```

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (lexer.py and preproc.py).
*   **Lexer:** Tokenizes the input code (lexer.py, tokens.py, and token\_kinds.py).
*   **Parser:** Uses recursive descent to parse the code and build a parse tree (parser/\*.py, tree/\*.py).
*   **IL Generation:** Transforms the parse tree into a custom intermediate language (il\_cmds/\*.py, il\_gen.py, tree/\*.py).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code, including register allocation (asm\_gen.py, il\_cmds/\*.py).

## Contributing

Please note that the project is no longer under active development.  Questions can be asked via Github Issues, and suggestions are welcome.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler that ShivyC is based on.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)