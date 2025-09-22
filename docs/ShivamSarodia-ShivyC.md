# ShivyC: A C Compiler Written in Python

**ShivyC** is a hobby-built C compiler designed to translate C code into efficient binaries, offering a fascinating look into compiler construction.  Check out the original repository at [https://github.com/ShivamSarodia/ShivyC](https://github.com/ShivamSarodia/ShivyC).

## Key Features

*   **C11 Subset Support:** ShivyC supports a significant portion of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries with built-in optimizations.
*   **Helpful Error Messages:** Provides informative compile-time error messages to assist with debugging.
*   **Written in Python:**  Leverages the versatility of Python for compiler design.

## Quickstart

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example Usage

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

### Running Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker Environment (for non-Linux users)

For users not on Linux, a Dockerfile is available.

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker setup:

    ```bash
    docker/shell
    ```

    Inside the Docker shell, you can compile and run:

    ```bash
    shivyc any_c_file.c
    python3 -m unittest discover
    ```

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives. (Implemented in `lexer.py` and `preproc.py`)
*   **Lexer:** Tokenizes the input C code. (Implemented in `lexer.py`, `tokens.py`, and `token_kinds.py`)
*   **Parser:**  Uses recursive descent to create a parse tree. (Implemented in `parser/*.py` and `tree/*.py`)
*   **IL Generation:**  Transforms the parse tree into a custom Intermediate Language (IL).  (Implemented in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`)
*   **ASM Generation:** Converts the IL into x86-64 assembly code. Includes register allocation using George and Appel's algorithm. (Implemented in `asm_gen.py` and `il_cmds/*.py`)

## Contributing

While active development has paused, contributions and questions are welcome via Github Issues.

## References

*   [ShivC (Original Compiler)](https://github.com/ShivamSarodia/ShivC)
*   C11 Specification: [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI: [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel): [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)