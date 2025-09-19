# angr: The Powerful Binary Analysis Framework

[![Latest Release](https://img.shields.io/pypi/v/angr.svg)](https://pypi.python.org/pypi/angr/)
[![Python Version](https://img.shields.io/pypi/pyversions/angr)](https://pypi.python.org/pypi/angr/)
[![PyPI Statistics](https://img.shields.io/pypi/dm/angr.svg)](https://pypistats.org/packages/angr)
[![License](https://img.shields.io/github/license/angr/angr.svg)](https://github.com/angr/angr/blob/master/LICENSE)

**angr empowers security researchers and developers with a versatile platform for advanced binary analysis.** Developed by the [Computer Security Lab at UC Santa Barbara](https://seclab.cs.ucsb.edu), [SEFCOM at Arizona State University](https://sefcom.asu.edu), [Shellphish](https://shellphish.net), and the open-source community.

[View the original repository on GitHub](https://github.com/angr/angr)

## Key Features

angr offers a comprehensive suite of tools for in-depth binary analysis:

*   **Disassembly and Intermediate Representation (IR) Lifting:**  Transforms binary code into a more manageable representation for analysis.
*   **Program Instrumentation:**  Allows for dynamic modification and observation of program behavior.
*   **Symbolic Execution:**  Enables exploration of all possible execution paths to uncover vulnerabilities.
*   **Control-Flow Analysis:**  Analyzes the structure and behavior of a program's execution flow.
*   **Data-Dependency Analysis:**  Identifies relationships between data values within a program.
*   **Value-Set Analysis (VSA):**  Provides precise information about possible values of variables.
*   **Decompilation:**  Translates compiled code back into a higher-level, more readable format.

## Getting Started

The core of angr is loading a binary. For example:  `p = angr.Project('/bin/bash')`

### Installation

Install angr with a virtual environment:

```bash
mkvirtualenv --python=$(which python3) angr && python -m pip install angr
```

### Examples

Here's a basic example demonstrating symbolic execution for a CTF challenge:

```python
import angr

project = angr.Project("angr-doc/examples/defcamp_r100/r100", auto_load_libs=False)

@project.hook(0x400844)
def print_flag(state):
    print("FLAG SHOULD BE:", state.posix.dumps(0))
    project.terminate_execution()

project.execute()
```

## Resources

*   **Homepage:** [https://angr.io](https://angr.io)
*   **Project Repository:** [https://github.com/angr/angr](https://github.com/angr/angr)
*   **Documentation:** [https://docs.angr.io](https://docs.angr.io)
*   **API Documentation:** [https://api.angr.io/en/latest/](https://api.angr.io/en/latest/)
*   **Install Instructions:** [https://docs.angr.io/introductory-errata/install](https://docs.angr.io/introductory-errata/install)
*   **Examples:** [https://docs.angr.io/examples](https://docs.angr.io/examples)
*   **Awesome-angr Repository:** [https://github.com/degrigis/awesome-angr](https://github.com/degrigis/awesome-angr)