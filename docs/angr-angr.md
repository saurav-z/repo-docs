# angr: Your Go-To Binary Analysis Framework for Security Research

angr is a powerful, platform-agnostic binary analysis framework, empowering security researchers and reverse engineers to dissect and understand software.  

[Link to Original Repository: https://github.com/angr/angr](https://github.com/angr/angr)

[![Latest Release](https://img.shields.io/pypi/v/angr.svg)](https://pypi.python.org/pypi/angr/)
[![Python Version](https://img.shields.io/pypi/pyversions/angr)](https://pypi.python.org/pypi/angr/)
[![PyPI Statistics](https://img.shields.io/pypi/dm/angr.svg)](https://pypistats.org/packages/angr)
[![License](https://img.shields.io/github/license/angr/angr.svg)](https://github.com/angr/angr/blob/master/LICENSE)

Developed by [the Computer Security Lab at UC Santa Barbara](https://seclab.cs.ucsb.edu), [SEFCOM at Arizona State University](https://sefcom.asu.edu), [Shellphish](https://shellphish.net), the open source community, and [@rhelmot](https://github.com/rhelmot).

## Key Features of angr

angr provides a comprehensive suite of tools for binary analysis, including:

*   **Disassembly and Intermediate Representation (IR) Lifting:**  Convert binary code into a more analyzable form.
*   **Program Instrumentation:**  Dynamically modify and monitor program behavior.
*   **Symbolic Execution:** Explore all possible execution paths to find vulnerabilities.
*   **Control-Flow Analysis:** Understand the program's structure and execution flow.
*   **Data-Dependency Analysis:** Identify how data influences program behavior.
*   **Value-Set Analysis (VSA):** Determine the possible values of variables.
*   **Decompilation:**  Translate binary code back into a higher-level representation.

## Getting Started with angr

angr is designed to be easy to use. The core operation involves loading a binary: `p = angr.Project('/bin/bash')`.  You can then explore the functionality using tab-autocomplete in an enhanced REPL like IPython.

**Installation:**

Install using: `mkvirtualenv --python=$(which python3) angr && python -m pip install angr`

## Example: Solving a CTF Challenge

Here's a simple example of using symbolic execution to find a flag:

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
*   **Examples:**  [https://docs.angr.io/examples](https://docs.angr.io/examples)
*   **Awesome angr:** [https://github.com/degrigis/awesome-angr](https://github.com/degrigis/awesome-angr)