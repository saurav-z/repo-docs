# ShivyC: A C Compiler in Python

**ShivyC is a hobby C compiler written in Python that supports a subset of the C11 standard, offering a glimpse into the world of compiler design.**  Check out the original repo [here](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** Implements a portion of the C11 standard.
*   **Python-Based:** Built entirely in Python 3.
*   **x86-64 Binary Generation:** Creates reasonably efficient binaries for x86-64 Linux.
*   **Optimization:** Includes basic optimizations for improved code performance.
*   **Helpful Error Messages:** Provides clear and informative compile-time error messages.
*   **Intermediate Language (IL):** Generates a custom IL before assembly code.
*   **Register Allocation:** Employs George and Appel's iterated register coalescing algorithm.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Compile and Run a "Hello, World!" Program

1.  Create a `hello.c` file:

    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  Compile the code:

    ```bash
    shivyc hello.c
    ```

3.  Run the compiled program:

    ```bash
    ./out
    ```

### Running Tests

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the tests:

    ```bash
    python3 -m unittest discover
    ```

### Docker for Other Architectures

For those not on Linux, use the provided Dockerfile to set up a development environment.

1.  Clone the repository (if you haven't already):

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker shell:

    ```bash
    docker/shell
    ```

    This opens a shell with ShivyC installed.  You can then compile and run tests within the container.

## Implementation Overview

*   **Preprocessor:**  Handles comments and `#include` directives, implemented in `lexer.py` and `preproc.py`.
*   **Lexer:**  Tokenizes the input source code, primarily in `lexer.py`, with token definitions in `tokens.py` and `token_kinds.py`.
*   **Parser:**  Uses recursive descent to create a parse tree in `parser/*.py`, using nodes defined in `tree/*.py`.
*   **IL Generation:**  Traverses the parse tree to generate a custom IL. The commands are in `il_cmds/*.py`. Most code is in `make_code` functions within `tree/*.py`.
*   **ASM Generation:** Converts IL commands into x86-64 assembly code (Intel format). Register allocation uses George and Appel's algorithm. Code is primarily within `asm_gen.py` and in `make_asm` functions within `il_cmds/*.py`.

## Contributing

While the project is no longer under active development, questions via GitHub Issues are welcome.

## References

*   **[ShivyC's predecessor, ShivC](https://github.com/ShivamSarodia/ShivC)**
*   C11 Specification: [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI: [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel): [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)