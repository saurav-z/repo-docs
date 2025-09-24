# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler written in Python that brings you closer to the inner workings of programming.** ([Original Repository](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

ShivyC is a C compiler implemented in Python 3, supporting a subset of the C11 standard. It's designed as a learning tool, generating reasonably efficient x86-64 binaries and providing helpful compile-time error messages. This project showcases the compilation process from source code to executable.

**Key Features:**

*   **C11 Subset Support:** Implements a functional subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries with some optimizations.
*   **Compile-Time Error Messages:** Provides helpful error messages to assist with debugging.
*   **Written in Python:** Built entirely in Python 3, making it accessible and easier to understand for Python developers.
*   **Educational Resource:** Great for understanding the compilation process and compiler design.

## Getting Started

### Prerequisites

*   Python 3.6 or later
*   GNU binutils and glibc (typically pre-installed on Linux)

### Installation

Install ShivyC using pip:

```bash
pip3 install shivyc
```

### Example: Compile and Run

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

### Testing

To run the tests, clone the repository and use the unittest module:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
python3 -m unittest discover
```

### Docker (for Other Architectures)

For convenience, a Dockerfile is provided to set up an x86-64 Ubuntu environment.

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
docker/shell
```

Inside the Docker shell, you can compile and test as above, and any changes you make locally will be reflected.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives.
*   **Lexer:** Converts source code into tokens.
*   **Parser:** Uses recursive descent to build a parse tree.
*   **IL Generation:** Creates a custom intermediate language.
*   **ASM Generation:** Converts IL to x86-64 assembly code, including register allocation using George and Appel's iterated register coalescing algorithm.

## Contributing

The project is not under active development, but issues and contributions are welcome.

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC) - The original C compiler.
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)