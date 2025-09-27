# angr: Powerful Binary Analysis Framework

[![Latest Release](https://img.shields.io/pypi/v/angr.svg)](https://pypi.python.org/pypi/angr/)
[![Python Version](https://img.shields.io/pypi/pyversions/angr)](https://pypi.python.org/pypi/angr/)
[![PyPI Statistics](https://img.shields.io/pypi/dm/angr.svg)](https://pypistats.org/packages/angr)
[![License](https://img.shields.io/github/license/angr/angr.svg)](https://github.com/angr/angr/blob/master/LICENSE)

**angr is the ultimate platform-agnostic binary analysis framework, empowering you to dissect and understand software like never before.** Developed by the Computer Security Lab at UC Santa Barbara, SEFCOM at Arizona State University, Shellphish, and the open-source community, angr provides a comprehensive suite of tools for reverse engineering, vulnerability research, and security analysis.

[Visit the original repo on GitHub](https://github.com/angr/angr)

## Key Features of angr

angr offers a wide array of features to facilitate in-depth binary analysis:

*   **Disassembly and Intermediate Representation (IR) Lifting:** Convert binary code into a more manageable format for analysis.
*   **Program Instrumentation:** Insert probes and modify program behavior to gather crucial insights.
*   **Symbolic Execution:** Explore all possible execution paths to uncover hidden vulnerabilities and logical flaws.
*   **Control-Flow Analysis:** Understand the program's structure and execution flow to identify potential security risks.
*   **Data-Dependency Analysis:** Trace how data flows through the program to pinpoint sources of sensitive information.
*   **Value-Set Analysis (VSA):** Reason about the possible values of variables and memory locations.
*   **Decompilation:** Reconstruct high-level code from low-level binaries to improve understanding.

## Getting Started with angr

The core operation involves loading a binary: `p = angr.Project('/bin/bash')`. Use tab-autocomplete in an enhanced REPL to explore available methods and docstrings.

Install with: `mkvirtualenv --python=$(which python3) angr && python -m pip install angr`

## Example: Solving a CTF Challenge

Here's a simple example demonstrating symbolic execution to extract a flag:

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
*   **Awesome-angr:** [https://github.com/degrigis/awesome-angr](https://github.com/degrigis/awesome-angr)