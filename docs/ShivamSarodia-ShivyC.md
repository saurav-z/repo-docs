# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built in Python, aiming to support a subset of the C11 standard and generate efficient x86-64 binaries.** Learn more about ShivyC on [GitHub](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** ShivyC implements a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries with optimizations.
*   **Helpful Error Messages:** Provides compile-time error messages to aid in debugging.
*   **Python-Based:** Written entirely in Python 3, making it easy to understand and extend.
*   **Includes a Trie Implementation:**  See this [implementation of a trie](tests/general_tests/trie/trie.c) as an example of what ShivyC can compile today.

## Quickstart

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation
```bash
pip3 install shivyc
```

### Example: Compile and Run

1.  Create `hello.c`:
    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```
2.  Compile:
    ```bash
    shivyc hello.c
    ```
3.  Run:
    ```bash
    ./out
    ```

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Using Docker (for non-Linux users)

For those not running Linux, the [`docker/`](docker/) directory provides a Dockerfile with an x86-64 Ubuntu environment pre-configured with everything needed for ShivyC.

1.  Build and run the Docker container:
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    docker/shell
    ```
2.  Inside the Docker shell, compile and run:
    ```bash
    shivyc any_c_file.c           # Compile a file
    python3 -m unittest discover  # Run tests
    ```
    Changes made to the local ShivyC directory are reflected live inside the Docker container.

## Implementation Overview

ShivyC's compilation process is broken down into several stages:

*   **Preprocessor:** Handles comments and `#include` directives ( [`lexer.py`](shivyc/lexer.py) and [`preproc.py`](shivyc/lexer.py) ).
*   **Lexer:** Transforms source code into tokens ( [`lexer.py`](shivyc/lexer.py), [`tokens.py`](shivyc/tokens.py), [`token_kinds.py`](shivyc/token_kinds.py) ).
*   **Parser:** Utilizes recursive descent parsing to create a parse tree ( [`parser/*.py`](shivyc/parser/) and [`tree/*.py`](shivyc/tree/) ).
*   **IL Generation:** Generates a custom intermediate language (IL) from the parse tree ( [`il_cmds/*.py`](shivyc/il_cmds/), [`il_gen.py`](shivyc/il_gen.py), and `make_code` functions within [`tree/*.py`](shivyc/tree/) ).
*   **ASM Generation:** Converts IL commands into Intel-format x86-64 assembly code, including register allocation using the George and Appel's iterated register coalescing algorithm ( [`asm_gen.py`](shivyc/asm_gen.py) and `make_asm` functions within [`il_cmds/*.py`](shivyc/il_cmds/) ).

## Contributing

This project is no longer under active development. However, feel free to:

*   Submit questions via GitHub Issues.
*   Suggest ideas for how ShivyC could be improved.

## References

*   [ShivC (Original Compiler)](https://github.com/ShivamSarodia/ShivC)
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)