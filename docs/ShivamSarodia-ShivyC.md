# ShivyC: A C Compiler in Python

**ShivyC is a hobby C compiler written in Python 3, offering a hands-on learning experience and supporting a subset of the C11 standard.** Explore the inner workings of a compiler and experiment with C code compilation!

[View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** Compiles a subset of the C11 standard, allowing you to experiment with common C language features.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries, including some optimization techniques.
*   **Informative Error Messages:** Provides helpful compile-time error messages to assist in debugging your C code.
*   **Python-Based:** Written entirely in Python 3, making it accessible and easier to understand for those familiar with Python.
*   **Educational:** Excellent for learning about compiler design, intermediate languages, and assembly generation.

## Quickstart

### Prerequisites
*   Python 3.6 or later
*   GNU binutils and glibc (usually pre-installed on Linux)

### Installation
```bash
pip3 install shivyc
```

### Example Usage
Create a `hello.c` file with the following content:

```c
#include <stdio.h>
int main() {
  printf("hello, world!\n");
}
```

Compile and run:

```bash
shivyc hello.c
./out
# Output: hello, world!
```

### Running Tests
```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker for Other Architectures
For a convenient development environment:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
# Inside the Docker shell:
shivyc any_c_file.c           # Compile a C file
python3 -m unittest discover  # Run tests
```

## Implementation Overview

ShivyC's compilation process is broken down into the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (`lexer.py` and `preproc.py`).
*   **Lexer:** Converts source code into tokens (`lexer.py`, `tokens.py`, `token_kinds.py`).
*   **Parser:** Constructs a parse tree using recursive descent techniques (`parser/*.py`, `tree/*.py`).
*   **Intermediate Language (IL) Generation:** Creates a custom flat IL from the parse tree (`il_cmds/*.py`, `il_gen.py`, `tree/*.py`).
*   **Assembly (ASM) Generation:** Transforms IL into x86-64 assembly code, using register allocation based on George and Appel’s iterated register coalescing algorithm (`asm_gen.py`, `il_cmds/*.py`).

## Contributing

While active development is limited, contributions and discussions are welcome.

*   **Questions:** Ask questions via GitHub Issues.
*   **Suggestions:** Propose ideas for practical improvements via GitHub Issues.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler that ShivyC is based on.
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf