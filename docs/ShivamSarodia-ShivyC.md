# ShivyC: A C Compiler Written in Python

**ShivyC** is a hobby C compiler written in Python that translates C code into efficient x86-64 binaries, providing helpful compile-time error messages.  Get started with ShivyC and explore its features on [GitHub](https://github.com/ShivamSarodia/ShivyC)!

## Key Features

*   **C11 Standard Support:** ShivyC supports a subset of the C11 standard.
*   **Optimized Binaries:** Generates reasonably efficient x86-64 binaries with optimizations.
*   **Informative Error Messages:** Provides helpful compile-time error messages to aid development.
*   **Written in Python:** Entirely implemented in Python 3.
*   **x86-64 Linux and Docker Support:** Ready to use on Linux and provided a Docker environment for ease of use.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically already installed on Linux)

### Installation

```bash
pip3 install shivyc
```

### Example: Compile and Run a "Hello, World!" Program

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

### Run Tests

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker Quickstart

For those not on Linux, use Docker for a ready-to-go environment:

1.  Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  Run the Docker shell:

```bash
docker/shell
```

Inside the Docker container, you can use `shivyc` and run the tests as described above.

## Implementation Overview

ShivyC's structure includes:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in `lexer.py` and `preproc.py`).
*   **Lexer:** Breaks down the source code into tokens (`lexer.py`, `tokens.py`, `token_kinds.py`).
*   **Parser:** Uses recursive descent to create a parse tree (`parser/*.py`, `tree/*.py`).
*   **IL Generation:** Generates a custom intermediate language (`il_cmds/*.py`, `il_gen.py`, and `tree/*.py`).
*   **ASM Generation:** Converts IL commands into x86-64 assembly code, including register allocation using George and Appel's algorithm (`asm_gen.py`, `il_cmds/*.py`).

## Contributing

This project is no longer actively developed. You can still submit questions via GitHub Issues, and I'm open to perspectives on making ShivyC practically useful to a group (please make an Issue).

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC): Original C compiler that ShivyC is based on.
*   C11 Specification: [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI: [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel): [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)