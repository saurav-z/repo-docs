# ShivyC: A C Compiler Written in Python

**ShivyC** is a hobby C compiler, written in Python, that allows you to compile C code and generate efficient binaries. [Check out the original repository here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:** Compiles a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably optimized x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages.
*   **Written in Python:** Leverages the versatility and readability of Python 3.
*   **Includes Optimizations:** Performs optimizations to improve performance.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically already installed on Linux)

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

2.  **Compile:**

    ```bash
    shivyc hello.c
    ```

3.  **Run:**

    ```bash
    ./out
    ```

### Running Tests

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  **Run the tests:**

    ```bash
    python3 -m unittest discover
    ```

### Docker for Other Architectures

For convenience, a Dockerfile is provided to set up an x86-64 Linux Ubuntu environment:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  **Run the Docker shell:**

    ```bash
    docker/shell
    ```

    Inside the Docker shell, you can compile and run tests:

    ```bash
    shivyc any_c_file.c           # Compile a file
    python3 -m unittest discover  # Run tests
    ```

    The Docker executable will update with any local changes.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (implemented in [`lexer.py`](shivyc/lexer.py) and [`preproc.py`](shivyc/lexer.py)).
*   **Lexer:** Converts source code into tokens (implemented primarily in [`lexer.py`](shivyc/lexer.py) and using definitions from [`tokens.py`](shivyc/tokens.py) and [`token_kinds.py`](shivyc/token_kinds.py)).
*   **Parser:** Uses recursive descent to create a parse tree (in [`parser/*.py`](shivyc/parser/) using nodes defined in [`tree/*.py`](shivyc/tree/)).
*   **IL Generation:** Transforms the parse tree into a custom intermediate language (IL) (commands in [`il_cmds/*.py`](shivyc/il_cmds/), objects in [`il_gen.py`](shivyc/il_gen.py), and `make_code` in [`tree/*.py`](shivyc/tree/)).
*   **ASM Generation:** Converts IL commands into x86-64 assembly (in [`asm_gen.py`](shivyc/asm_gen.py) and `make_asm` in [`il_cmds/*.py`](shivyc/il_cmds/), utilizing George and Appel’s iterated register coalescing algorithm).

## Contributing

*   Please direct questions through Github Issues.
*   Suggest improvements or new features via Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf