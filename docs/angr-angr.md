# angr: The Powerful Binary Analysis Framework for Security Research and CTF Challenges

Unleash the power of symbolic execution and binary analysis with angr, a cutting-edge platform-agnostic framework. Explore the original repository [here](https://github.com/angr/angr).

[![Latest Release](https://img.shields.io/pypi/v/angr.svg)](https://pypi.python.org/pypi/angr/)
[![Python Version](https://img.shields.io/pypi/pyversions/angr)](https://pypi.python.org/pypi/angr/)
[![PyPI Statistics](https://img.shields.io/pypi/dm/angr.svg)](https://pypistats.org/packages/angr)
[![License](https://img.shields.io/github/license/angr/angr.svg)](https://github.com/angr/angr/blob/master/LICENSE)

**angr** is a suite of Python 3 libraries designed for advanced binary analysis. It's developed by the [Computer Security Lab at UC Santa Barbara](https://seclab.cs.ucsb.edu), [SEFCOM at Arizona State University](https://sefcom.asu.edu), their associated CTF team, [Shellphish](https://shellphish.net), the open source community, and **[@rhelmot](https://github.com/rhelmot)**.

## Key Features of angr:

*   **Platform-Agnostic Binary Analysis:** Analyze binaries across different architectures.
*   **Disassembly and Intermediate Representation (IR) Lifting:** Converts binary code into an understandable intermediate representation.
*   **Program Instrumentation:** Modify and monitor the execution of your programs.
*   **Symbolic Execution:** Explore all possible execution paths of a program.
*   **Control-Flow Analysis:** Understand how a program's execution flows.
*   **Data-Dependency Analysis:** Identify relationships between data within the program.
*   **Value-Set Analysis (VSA):**  Determine the possible values of variables.
*   **Decompilation:**  Convert machine code back into a higher-level representation.

## Getting Started with angr

### Installation

Install angr with Python's `pip` package manager:
```bash
mkvirtualenv --python=$(which python3) angr && python -m pip install angr
```

### Basic Usage

Load a binary using `angr.Project()`:

```python
import angr
project = angr.Project('/bin/bash')
```

## Example: Solving a CTF Challenge

Here's a simple example demonstrating how to use symbolic execution to retrieve a flag in a CTF challenge:

```python
import angr

project = angr.Project("angr-doc/examples/defcamp_r100/r100", auto_load_libs=False)

@project.hook(0x400844)
def print_flag(state):
    print("FLAG SHOULD BE:", state.posix.dumps(0))
    project.terminate_execution()

project.execute()
```

## Useful Resources

*   **Homepage:** [https://angr.io](https://angr.io)
*   **Project Repository:** [https://github.com/angr/angr](https://github.com/angr/angr)
*   **Documentation:** [https://docs.angr.io](https://docs.angr.io)
*   **API Documentation:** [https://api.angr.io/en/latest/](https://api.angr.io/en/latest/)
*   **Install Instructions:** [https://docs.angr.io/introductory-errata/install](https://docs.angr.io/introductory-errata/install)
*   **Examples:** [https://docs.angr.io/examples](https://docs.angr.io/examples)
*   **Awesome-angr:** [https://github.com/degrigis/awesome-angr](https://github.com/degrigis/awesome-angr)