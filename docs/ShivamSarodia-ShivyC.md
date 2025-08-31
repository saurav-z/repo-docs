# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built in Python, allowing you to compile a subset of the C11 standard and experience the inner workings of a compiler.**  Explore the inner workings of a compiler with ShivyC, a C compiler written in Python.

[View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Standard Support:** Supports a subset of the C11 standard.
*   **Python-Based:** Written entirely in Python 3, making it easy to understand and modify.
*   **x86-64 Assembly Generation:** Generates efficient x86-64 assembly code.
*   **Compile-Time Error Messages:** Provides helpful error messages to aid in debugging.
*   **Includes Optimizations:** Implements optimizations to improve generated code performance.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Example: Compile and Run

1.  **Create a `hello.c` file:**

    ```c
    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```

2.  **Compile and run the code:**

    ```bash
    shivyc hello.c
    ./out
    ```

### Testing

Clone the repository and run the tests:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Using Docker

For non-Linux users, the `docker/` directory provides a Dockerfile for an x86-64 Linux Ubuntu environment pre-configured with ShivyC.

1.  **Build and run the Docker container:**

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    docker/shell
    ```

2.  **Inside the Docker container, compile and test:**

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives (lexer.py, preproc.py).
*   **Lexer:** Tokenizes the input code (lexer.py, tokens.py, token\_kinds.py).
*   **Parser:** Uses recursive descent to create a parse tree (parser/\*.py, tree/\*.py).
*   **Intermediate Language (IL) Generation:** Translates the parse tree into a custom IL (il\_cmds/\*.py, il\_gen.py, tree/\*.py).
*   **Assembly Generation:** Converts IL commands into x86-64 assembly code, including register allocation using George and Appel's iterated register coalescing algorithm (asm\_gen.py, il\_cmds/\*.py).

## Contributing

**Note:** This project is no longer under active development.

*   For questions, use GitHub Issues.
*   For suggestions on making ShivyC practically helpful, create an Issue.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC): The original C compiler this project is based on.
*   C11 Specification: [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI: [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel): [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)