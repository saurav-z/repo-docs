# angr: The Powerful Binary Analysis Framework for Security Research & Reverse Engineering

[![Latest Release](https://img.shields.io/pypi/v/angr.svg)](https://pypi.python.org/pypi/angr/)
[![Python Version](https://img.shields.io/pypi/pyversions/angr)](https://pypi.python.org/pypi/angr/)
[![PyPI Statistics](https://img.shields.io/pypi/dm/angr.svg)](https://pypistats.org/packages/angr)
[![License](https://img.shields.io/github/license/angr/angr.svg)](https://github.com/angr/angr/blob/master/LICENSE)

**angr is a cutting-edge, platform-agnostic binary analysis framework, empowering security researchers and reverse engineers to understand and analyze software at a deeper level.** Developed by the Computer Security Lab at UC Santa Barbara, SEFCOM at Arizona State University, and the open-source community, angr provides a comprehensive suite of tools for dissecting and understanding compiled code.

**[View the original repository on GitHub](https://github.com/angr/angr)**

## Key Features of angr

*   **Disassembly and Intermediate Representation (IR) Lifting:** Converts binary code into a more manageable and analyzable form.
*   **Program Instrumentation:** Allows for dynamic analysis and modification of program behavior.
*   **Symbolic Execution:** Enables exploration of all possible execution paths to identify vulnerabilities.
*   **Control-Flow Analysis:** Provides insights into the program's structure and how different parts of the code interact.
*   **Data-Dependency Analysis:** Helps understand how data flows through the program.
*   **Value-Set Analysis (VSA):** Determines the possible values a variable can take during program execution.
*   **Decompilation:** Attempts to convert machine code back into a higher-level representation.

## Getting Started with angr

The most fundamental operation in angr is loading a binary: `p = angr.Project('/bin/bash')`. Leverage enhanced REPLs like IPython for easy exploration and discovery.

### Installation

The recommended way to install angr is:
```bash
mkvirtualenv --python=$(which python3) angr && python -m pip install angr
```

### Example: Solving a CTF Challenge

Here's a basic illustration of how to use symbolic execution with angr to solve a CTF challenge:

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
*   **Documentation:** [https://docs.angr.io](https://docs.angr.io)
*   **API Documentation:** [https://api.angr.io/en/latest/](https://api.angr.io/en/latest/)
*   **Install Instructions:** [https://docs.angr.io/introductory-errata/install](https://docs.angr.io/introductory-errata/install)
*   **Examples:** [https://docs.angr.io/examples](https://docs.angr.io/examples)
*   **Awesome angr:** [https://github.com/degrigis/awesome-angr](https://github.com/degrigis/awesome-angr)