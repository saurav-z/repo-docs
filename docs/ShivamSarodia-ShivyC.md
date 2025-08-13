# ShivyC: A Hobby C Compiler in Python

**ShivyC is a C compiler written in Python that supports a subset of the C11 standard, generating efficient binaries with helpful compile-time error messages.** This project, available on [GitHub](https://github.com/ShivamSarodia/ShivyC), is a great way to learn about compilers or experiment with C code.

## Key Features

*   **C11 Subset Support:** Compiles a significant portion of the C11 standard.
*   **Optimized Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Clear Error Messages:** Provides helpful compile-time error messages for easier debugging.
*   **Written in Python 3:** Easy to understand and modify for educational purposes.

## Quickstart

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc (typically already installed on Linux systems)

### Installation
```bash
pip3 install shivyc
```

### Example Usage

Create a `hello.c` file:

```c
// hello.c
#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```

Compile and run:

```bash
shivyc hello.c
./out
```

## Other Architectures (with Docker)

For ease of use, a Dockerfile is provided in the `docker/` directory. This sets up an x86-64 Linux Ubuntu environment with everything needed for ShivyC.

To use Docker:

1.  Clone the repository:
    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```
2.  Run the Docker shell:
    ```bash
    docker/shell
    ```

Inside the Docker environment, you can compile and run C files:

```bash
shivyc any_c_file.c
```

Run tests:

```bash
python3 -m unittest discover
```

## Implementation Overview

ShivyC's architecture includes the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (in `lexer.py` and `preproc.py`).
*   **Lexer:** Tokenizes the input C code (in `lexer.py`, `tokens.py`, and `token_kinds.py`).
*   **Parser:** Uses recursive descent to build a parse tree (in `parser/*.py` and `tree/*.py`).
*   **Intermediate Language (IL) Generation:** Translates the parse tree into a custom IL (in `il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **Assembly (ASM) Generation:** Converts the IL into Intel x86-64 assembly code (in `asm_gen.py` and `il_cmds/*.py`), including register allocation using George and Appel's iterated register coalescing.

## Contributing

Please note that the project is not under active development. However, feel free to open an issue on GitHub with questions or suggestions.

## References

*   [ShivyC Repository](https://github.com/ShivamSarodia/ShivyC)
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)