# ShivyC: A C Compiler Written in Python

**ShivyC is a hobby C compiler built in Python 3 that brings a subset of the C11 standard to life.** ([Original Repo](https://github.com/ShivamSarodia/ShivyC))

[![Build Status](https://travis-ci.org/ShivamSarodia/ShivyC.svg?branch=master)](https://travis-ci.org/ShivamSarodia/ShivyC)
[![Code Coverage](https://codecov.io/gh/ShivamSarodia/ShivyC/branch/master/graph/badge.svg)](https://codecov.io/gh/ShivamSarodia/ShivyC)

<img src="https://raw.githubusercontent.com/ShivamSarodia/ShivyC/master/demo.gif" alt="ShivyC demo GIF" width="500"/>

## Key Features

*   **C11 Standard Support:** Implements a subset of the C11 standard.
*   **Efficient Binaries:** Generates reasonably efficient x86-64 binaries.
*   **Compile-Time Error Messages:** Provides helpful error messages to aid in debugging.
*   **Written in Python 3:** Easy to install and use with Python 3.6 or later.
*   **Includes Optimizations:** ShivyC implements optimization strategies.
*   **Docker Support:** Provides a Dockerfile for easy setup and use on various platforms.

## Quickstart

### Prerequisites

*   Python 3.6 or later
*   GNU binutils
*   glibc

### Installation

```bash
pip3 install shivyc
```

### Example Usage

1.  **Create a C file**

    ```c
    $ vim hello.c
    $ cat hello.c

    #include <stdio.h>
    int main() {
      printf("hello, world!\n");
    }
    ```
2.  **Compile and run**

    ```bash
    shivyc hello.c
    ./out
    ```

### Running Tests

1.  **Clone the repository**

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```
2.  **Run the tests**

    ```bash
    python3 -m unittest discover
    ```

### Docker Quickstart

For users not on Linux, a Dockerfile is provided for easy setup.

1.  **Clone the repository**

    ```bash
    git clone https://github.com/ShivamSarodia/ShivyC.git
    cd ShivyC
    ```
2.  **Build the Docker image**

    ```bash
    docker build -t shivyc docker/
    ```
3.  **Run the Docker container**

    ```bash
    docker/shell
    ```

    Inside the container, you can compile with `shivyc any_c_file.c` and run tests with `python3 -m unittest discover`.  Changes in the local ShivyC directory will update live in the Docker environment.

## Implementation Overview

*   **Preprocessor:** Handles comments and `#include` directives. (lexer.py, preproc.py)
*   **Lexer:** Breaks down the code into tokens. (lexer.py, tokens.py, token\_kinds.py)
*   **Parser:** Uses recursive descent to create a parse tree. (parser/\*.py, tree/nodes.py, tree/expr\_nodes.py)
*   **IL Generation:** Transforms the parse tree into a custom Intermediate Language. (il\_cmds/\*.py, il\_gen.py, tree/\*.py)
*   **ASM Generation:** Converts IL commands into x86-64 assembly code. (asm\_gen.py, il\_cmds/\*.py) with register allocation using George and Appel's iterated register coalescing algorithm.

## Contributing

Contributions are welcome! Check out the [Issues page](https://github.com/ShivamSarodia/ShivyC/issues) for feature requests and bug reports. Create a new issue labeled "question" if you have questions.  Please add tests for all new functionality.

**Contributors:**

*   ShivamSarodia
*   cclauss
*   TBladen
*   christian-stephen
*   jubnzv
*   eriols

## References

*   [ShivC](https://github.com/ShivamSarodia/ShivC)
*   C11 Specification - [http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
*   x86\_64 ABI - [https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf](https://github.com/hjl-tools/x86-psABI/wiki/x86-64-psABI-1.0.pdf)
*   Iterated Register Coalescing (George and Appel) - [https://www.cs.purdue.edu/homes/hosking/502/george.pdf](https://www.cs.purdue.edu/homes/hosking/502/george.pdf)