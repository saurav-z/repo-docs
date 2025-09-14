# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built in Python 3, offering a unique perspective on compiler design and functionality.**  [View the original repository on GitHub](https://github.com/ShivamSarodia/ShivyC)

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC) [![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC supports a subset of the C11 standard and generates reasonably efficient binaries, including some optimizations. It also provides helpful compile-time error messages.

## Key Features

*   **Written in Python:** ShivyC provides an accessible way to learn about compiler design using a familiar language.
*   **C11 Subset Support:** Compiles a portion of the C11 standard, allowing you to experiment with C code.
*   **Optimized Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Clear Error Messages:** Provides informative error messages to help with debugging.
*   **Educational Project:** A great resource for understanding compiler internals and C compilation processes.

## Quickstart

### x86-64 Linux

**Prerequisites:** Python 3.6 or later, GNU binutils, and glibc.

**Installation:**

```bash
pip3 install shivyc
```

**Example Usage:**

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
# Output: hello, world!
```

**Running Tests:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Other Architectures (using Docker)

The `docker/` directory provides a Dockerfile to set up an x86-64 Linux Ubuntu environment with everything needed for ShivyC.

**Usage:**

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

This will open a shell with ShivyC installed.

*   Compile a C file: `shivyc any_c_file.c`
*   Run tests: `python3 -m unittest discover`

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives.
*   **Lexer:** Converts source code into tokens.
*   **Parser:** Utilizes recursive descent parsing to build a parse tree.
*   **IL Generation:** Creates a custom intermediate language.
*   **ASM Generation:** Translates the IL into x86-64 assembly code, including register allocation using George and Appel's iterated register coalescing algorithm.

## Contributing

This project is no longer under active development, but contributions and questions are welcome.

*   **Questions:**  Use Github Issues.
*   **Suggestions:**  Create an Issue to propose ideas for improvements.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler from which ShivyC was rewritten
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf