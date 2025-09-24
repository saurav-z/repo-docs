# angr: The Powerful Binary Analysis Framework 🕵️‍♀️

**angr is the ultimate platform-agnostic binary analysis framework, empowering you to dissect and understand compiled code.**  Get started with angr today by visiting the original repo: [https://github.com/angr/angr](https://github.com/angr/angr).

[![Latest Release](https://img.shields.io/pypi/v/angr.svg)](https://pypi.python.org/pypi/angr/)
[![Python Version](https://img.shields.io/pypi/pyversions/angr)](https://pypi.python.org/pypi/angr/)
[![PyPI Statistics](https://img.shields.io/pypi/dm/angr.svg)](https://pypistats.org/packages/angr)
[![License](https://img.shields.io/github/license/angr/angr.svg)](https://github.com/angr/angr/blob/master/LICENSE)

Brought to you by [the Computer Security Lab at UC Santa Barbara](https://seclab.cs.ucsb.edu), [SEFCOM at Arizona State University](https://sefcom.asu.edu), [Shellphish](https://shellphish.net), and the open source community.

## Key Features of angr

angr provides a comprehensive suite of tools for in-depth binary analysis, including:

*   **Disassembly and Intermediate Representation (IR) Lifting:**  Transforms machine code into a more manageable format.
*   **Program Instrumentation:**  Allows for dynamic analysis and modification of program behavior.
*   **Symbolic Execution:**  Explores all possible execution paths to uncover hidden vulnerabilities.
*   **Control-Flow Analysis (CFA):**  Maps out the program's structure to understand its flow.
*   **Data-Dependency Analysis:**  Tracks how data flows through the program.
*   **Value-Set Analysis (VSA):** Analyzes the possible values of variables and registers.
*   **Decompilation:**  Reverse-engineers machine code back into a more readable form.

## Getting Started with angr

The core of angr involves loading and analyzing a binary.  Here's a simple example:

```python
import angr

project = angr.Project("/bin/bash") # Replace with your binary
```

For installation, use:  `mkvirtualenv --python=$(which python3) angr && python -m pip install angr`.

## Useful Resources

*   **Documentation:** [https://docs.angr.io](https://docs.angr.io)
*   **API Documentation:** [https://api.angr.io/en/latest/](https://api.angr.io/en/latest/)
*   **Examples:** [https://docs.angr.io/examples](https://docs.angr.io/examples)
*   **Awesome angr:** [https://github.com/degrigis/awesome-angr](https://github.com/degrigis/awesome-angr)