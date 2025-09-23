# angr: The Powerful Binary Analysis Framework

**angr is a cutting-edge, platform-agnostic binary analysis framework that empowers security researchers and reverse engineers to explore and understand software behavior.** Developed by the Computer Security Lab at UC Santa Barbara, SEFCOM at Arizona State University, and the open-source community, angr provides a robust set of tools for in-depth binary analysis.  Find out more and contribute on the [original repository](https://github.com/angr/angr).

[![Latest Release](https://img.shields.io/pypi/v/angr.svg)](https://pypi.python.org/pypi/angr/)
[![Python Version](https://img.shields.io/pypi/pyversions/angr)](https://pypi.python.org/pypi/angr/)
[![PyPI Statistics](https://img.shields.io/pypi/dm/angr.svg)](https://pypistats.org/packages/angr)
[![License](https://img.shields.io/github/license/angr/angr.svg)](https://github.com/angr/angr/blob/master/LICENSE)

## Key Features of angr

angr offers a comprehensive suite of features for in-depth binary analysis, including:

*   **Disassembly and Intermediate Representation (IR) Lifting:** Transforms binary code into an intermediate representation for easier analysis.
*   **Program Instrumentation:** Allows for dynamic analysis by inserting probes and monitoring code execution.
*   **Symbolic Execution:** Explores all possible execution paths of a program, revealing hidden vulnerabilities and behaviors.
*   **Control-Flow Analysis (CFA):** Analyzes the program's control flow to understand its structure and identify potential issues.
*   **Data-Dependency Analysis:** Tracks how data flows through the program to uncover dependencies and potential security flaws.
*   **Value-Set Analysis (VSA):** Determines the possible values of variables to help understand program behavior.
*   **Decompilation:** Converts binary code back into a higher-level language representation.

## Getting Started with angr

angr is easy to get started with.  The core function is loading a binary: `p = angr.Project('/bin/bash')`

### Installation

Install angr using pip:

```bash
mkvirtualenv --python=$(which python3) angr && python -m pip install angr
```

### Project Links

*   **Homepage:** [https://angr.io](https://angr.io)
*   **Project Repository:** [https://github.com/angr/angr](https://github.com/angr/angr)
*   **Documentation:** [https://docs.angr.io](https://docs.angr.io)
*   **API Documentation:** [https://api.angr.io/en/latest/](https://api.angr.io/en/latest/)

### Quick Start Resources

*   **Install Instructions:** [https://docs.angr.io/introductory-errata/install](https://docs.angr.io/introductory-errata/install)
*   **Documentation:** [https://docs.angr.io](https://docs.angr.io)
*   **Top-Level Methods:** [https://docs.angr.io/core-concepts/toplevel](https://docs.angr.io/core-concepts/toplevel)
*   **Examples:** [https://docs.angr.io/examples](https://docs.angr.io/examples)
*   **API Reference:** [https://angr.io/api-doc/](https://angr.io/api-doc/)
*   **Awesome angr:** [https://github.com/degrigis/awesome-angr](https://github.com/degrigis/awesome-angr)

## Example: Symbolic Execution

Here's a simple example of using symbolic execution to get a flag in a CTF challenge:

```python
import angr

project = angr.Project("angr-doc/examples/defcamp_r100/r100", auto_load_libs=False)

@project.hook(0x400844)
def print_flag(state):
    print("FLAG SHOULD BE:", state.posix.dumps(0))
    project.terminate_execution()

project.execute()
```