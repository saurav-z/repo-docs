# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, built entirely in Python, offering a glimpse into the world of compiler design and functionality.**  [Explore the original repository](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC implements a subset of the C11 standard.
*   **Python-Based:** Written entirely in Python 3, making it accessible and easier to understand.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries, including some optimizations.
*   **Helpful Error Messages:** Provides clear and informative compile-time error messages.
*   **Includes example implementations**: See this [implementation of a trie](tests/general_tests/trie/trie.c) as an example of what ShivyC can compile today. For a more comprehensive list of features, see the [feature test directory](tests/feature_tests).

## Quickstart Guide

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation
```bash
pip3 install shivyc
```

### Compile and Run a "Hello, World" Program
```c
$ vim hello.c
$ cat hello.c

#include <stdio.h>
int main() {
  printf("hello, world!\n");
}

$ shivyc hello.c
$ ./out
hello, world!
```

### Running Tests
```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Using Docker (for non-Linux users or isolated environments)
1.  Clone the repository:
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```
2.  Run the Docker setup:
    ```bash
    docker/shell
    ```
    This will open a shell with ShivyC ready to use.

    *   Compile a C file: `shivyc any_c_file.c`
    *   Run tests: `python3 -m unittest discover`
    *   The Docker executable will update live with any changes in your local directory.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (lexer.py, preproc.py).
*   **Lexer:**  Tokenizes the source code (lexer.py, tokens.py, token_kinds.py).
*   **Parser:** Uses recursive descent for parsing and creates a parse tree (parser/*.py, tree/*.py).
*   **IL Generation:** Converts the parse tree into a custom Intermediate Language (IL) (il_cmds/*.py, il_gen.py, tree/*.py).
*   **ASM Generation:** Translates IL commands into x86-64 assembly code, with register allocation using George and Appel’s iterated register coalescing algorithm (asm_gen.py, il_cmds/*.py).

## Contributing

While active development has slowed, questions and suggestions are welcome:

*   **Questions:** Use Github Issues.
*   **Suggestions for Practical Use:** Submit an Issue with your ideas.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler rewritten to create ShivyC.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf