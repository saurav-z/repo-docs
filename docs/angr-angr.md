# angr: The Powerful Binary Analysis Framework

**Unlock the secrets of software with angr, a leading-edge, platform-agnostic binary analysis framework.**  Developed by the Computer Security Lab at UC Santa Barbara, SEFCOM at Arizona State University, and the open-source community, angr empowers you to dissect and understand binary code with ease. You can explore the original repository [here](https://github.com/angr/angr).

[![Latest Release](https://img.shields.io/pypi/v/angr.svg)](https://pypi.python.org/pypi/angr/)
[![Python Version](https://img.shields.io/pypi/pyversions/angr)](https://pypi.python.org/pypi/angr/)
[![PyPI Statistics](https://img.shields.io/pypi/dm/angr.svg)](https://pypistats.org/packages/angr)
[![License](https://img.shields.io/github/license/angr/angr.svg)](https://github.com/angr/angr/blob/master/LICENSE)

## Key Features of angr:

*   **Platform-Agnostic Binary Analysis:** Works across various architectures and operating systems.
*   **Disassembly and Intermediate Representation (IR) Lifting:** Convert binaries into a more manageable form for analysis.
*   **Program Instrumentation:**  Modify and monitor program behavior.
*   **Symbolic Execution:** Explore all possible execution paths to uncover vulnerabilities and find solutions.
*   **Control-Flow Analysis:**  Understand the program's structure and how it executes.
*   **Data-Dependency Analysis:** Trace data flow to identify critical information.
*   **Value-Set Analysis (VSA):** Precisely determine the range of possible values.
*   **Decompilation:** Convert binary code into higher-level representations to improve understanding.

## Getting Started with angr

Loading a binary is as simple as: `p = angr.Project('/bin/bash')`. Explore the API with tab-completion in an enhanced REPL (e.g., IPython).

**Installation:**

```bash
mkvirtualenv --python=$(which python3) angr && python -m pip install angr
```

## Example: Solving a CTF Challenge

```python
import angr

project = angr.Project("angr-doc/examples/defcamp_r100/r100", auto_load_libs=False)

@project.hook(0x400844)
def print_flag(state):
    print("FLAG SHOULD BE:", state.posix.dumps(0))
    project.terminate_execution()

project.execute()
```

## Project Links

*   **Homepage:** [https://angr.io](https://angr.io)
*   **Project Repository:** [https://github.com/angr/angr](https://github.com/angr/angr)
*   **Documentation:** [https://docs.angr.io](https://docs.angr.io)
*   **API Documentation:** [https://api.angr.io/en/latest/](https://api.angr.io/en/latest/)
*   **Install Instructions:** [https://docs.angr.io/introductory-errata/install](https://docs.angr.io/introductory-errata/install)
*   **Examples:** [https://docs.angr.io/examples](https://docs.angr.io/examples)
*   **API Reference:** [https://angr.io/api-doc/](https://angr.io/api-doc/)
*   **Awesome-angr repo:** [https://github.com/degrigis/awesome-angr](https://github.com/degrigis/awesome-angr)