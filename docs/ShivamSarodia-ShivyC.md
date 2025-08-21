# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler, built in Python, that aims to support a subset of the C11 standard and generate efficient binaries.** Check out the original repository at [https://github.com/ShivamSarodia/ShivyC](https://github.com/ShivamSarodia/ShivyC).

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

## Key Features

*   **C11 Subset Support:** Compiles a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Helpful Error Messages:** Provides informative compile-time error messages.
*   **Written in Python:**  Leverages Python 3 for its implementation.
*   **Includes Optimizations:** Includes optimizations for enhanced performance.

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

### Testing

1.  Clone the repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  Run tests:

```bash
python3 -m unittest discover
```

### Docker Environment

For users not on Linux, a Dockerfile is provided.

1.  Clone the repository:

```bash
git clone https://github.com/ShivamSarodia/ShivyC.git
cd ShivyC
```

2.  Run the Docker shell:

```bash
docker/shell
```

Inside the Docker shell, you can compile and test as described above. Changes to your local ShivyC directory are reflected live in the Docker environment.

## Implementation Overview

ShivyC's compilation process includes the following stages:

*   **Preprocessor:** Handles comments and `#include` directives (implemented in [`lexer.py`](shivyc/lexer.py) and [`preproc.py`](shivyc/lexer.py)).
*   **Lexer:** Breaks the source code into tokens (primarily in [`lexer.py`](shivyc/lexer.py), with token definitions in [`tokens.py`](shivyc/tokens.py) and [`token_kinds.py`](shivyc/token_kinds.py)).
*   **Parser:** Uses recursive descent techniques to create a parse tree (in [`parser/*.py`](shivyc/parser/) and [`tree/*.py`](shivyc/tree/)).
*   **IL Generation:**  Translates the parse tree into a custom intermediate language (IL) (in [`il_cmds/*.py`](shivyc/il_cmds/), [`il_gen.py`](shivyc/il_gen.py), and `make_code` functions in [`tree/*.py`](shivyc/tree/)).
*   **ASM Generation:**  Converts IL commands into x86-64 assembly code, including register allocation using the George and Appel iterated register coalescing algorithm (in [`asm_gen.py`](shivyc/asm_gen.py) and `make_asm` functions in [`il_cmds/*.py`](shivyc/il_cmds/)).

## Contributing

While active development has slowed, questions and suggestions are welcome via [Github Issues](https://github.com/ShivamSarodia/ShivyC/issues).

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC)
*   C11 Specification - http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
*   x86\_64 ABI - https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf
*   Iterated Register Coalescing (George and Appel) - https://www.cs.purdue.edu/homes/hosking/502/george.pdf