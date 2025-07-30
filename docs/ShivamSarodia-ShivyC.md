# ShivyC: A C Compiler in Python

ShivyC is a hobby C compiler written in Python 3 that aims to support a subset of the C11 standard and generate efficient binaries.  [Explore the project on GitHub](https://github.com/ShivamSarodia/ShivyC).

## Key Features

*   **C11 Subset Support:** Compiles a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** Provides clear and informative compile-time error messages.
*   **Written in Python:** Leverages the flexibility and readability of Python 3.
*   **Includes Optimizations:** Applies various optimizations to improve generated code.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example: Compiling and Running a Simple Program

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

### Docker for Other Architectures

For convenience, a Dockerfile is provided in the `docker/` directory to set up an x86-64 Linux Ubuntu environment.

1.  Clone the repository:

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```

2.  Run the Docker shell:

    ```bash
    docker/shell
    ```

    This opens a shell with ShivyC installed, ready to use:

    ```bash
    shivyc any_c_file.c           # to compile a file
    python3 -m unittest discover  # to run tests
    ```

    The Docker ShivyC executable updates live with any changes in your local directory.

## Implementation Overview

### Preprocessor

The preprocessor handles comments and `#include` directives. Implemented between [`lexer.py`](shivyc/lexer.py) and [`preproc.py`](shivyc/lexer.py).

### Lexer

The lexer converts source code into tokens.  Implemented primarily in [`lexer.py`](shivyc/lexer.py). Token definitions are in [`tokens.py`](shivyc/tokens.py) and token kinds are in [`token_kinds.py`](shivyc/token_kinds.py).

### Parser

The parser utilizes recursive descent techniques to create a parse tree. Located in [`parser/*.py`](shivyc/parser/) with parse tree nodes defined in [`tree/*.py`](shivyc/tree/).

### IL Generation

The parse tree is traversed to generate a custom IL (intermediate language).  IL commands are in [`il_cmds/*.py`](shivyc/il_cmds/), and IL generation utilizes objects in [`il_gen.py`](shivyc/il_gen.py). Most of the IL generating code is in the `make_code` function of each tree node in [`tree/*.py`](shivyc/tree/).

### ASM Generation

The IL commands are converted into Intel-format x86-64 assembly code. Register allocation uses George and Appel's iterated register coalescing algorithm.  General ASM generation is in [`asm_gen.py`](shivyc/asm_gen.py), with code primarily in the `make_asm` function of each IL command in [`il_cmds/*.py`](shivyc/il_cmds/).

## Contributing

*   **Issues:**  The best way to ask questions or suggest improvements is via GitHub Issues.
*   **Project Status:** Note the project is no longer under active development.  Contributions will be accepted but might not be reviewed.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler written by the author.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)