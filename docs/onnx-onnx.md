<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Open Neural Network Exchange (ONNX): Interoperability for AI Models

ONNX (Open Neural Network Exchange) is an open-source format that enables AI developers to seamlessly move between different frameworks and tools, fostering innovation and collaboration.  **[Explore the original repository](https://github.com/onnx/onnx).**

## Key Features

*   **Open Standard:** Defines a standard format for AI models, supporting both deep learning and traditional machine learning.
*   **Framework Interoperability:**  Allows you to convert and run models across various frameworks (e.g., PyTorch, TensorFlow, etc.).
*   **Extensible Computation Graph:** Provides a flexible graph model for representing AI computations.
*   **Operator Library:**  Includes a comprehensive set of built-in operators and standard data types.
*   **Wide Support:** Supported by numerous tools, frameworks, and hardware platforms.
*   **Focus on Inferencing:** Primarily designed to facilitate efficient model scoring and deployment.

## Getting Started

### Use ONNX

*   [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   [Pre-trained ONNX models](https://github.com/onnx/models)

### Learn about the ONNX Spec

*   [Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md)
*   [ONNX intermediate representation spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)
*   [Versioning principles of the spec](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
*   [Operators documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
*   [Operators documentation](https://onnx.ai/onnx/operators/index.html) (latest release)
*   [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)

### Programming Utilities for Working with ONNX Graphs

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

## Contributing

ONNX is a community-driven project, and we encourage your contributions!

*   **Community Governance:** Learn about the open governance model [here](https://github.com/onnx/onnx/blob/main/community/readme.md).
*   **Get Involved:** Participate in [Special Interest Groups](https://github.com/onnx/onnx/blob/main/community/sigs.md) and [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md).
*   **Contribution Guide:** Review the [contribution guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) to get started.
*   **Adding Operators:** If you want to add an operator, please read [this document](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md).

## Community

*   **Community Meetings:** Find the schedule of meetings [here](https://onnx.ai/calendar).
*   **Community Meetups:**  View content from past community meetups:
    *   2020.04.09 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14091402/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+2020+April+9>
    *   2020.10.14 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092138/LF+AI+Day+-+ONNX+Community+Workshop+-+2020+October+14>
    *   2021.03.24 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092424/Instructions+for+Event+Hosts+-+LF+AI+Data+Day+-+ONNX+Virtual+Community+Meetup+-+March+2021>
    *   2021.10.21 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093194/LF+AI+Data+Day+ONNX+Community+Virtual+Meetup+-+October+2021>
    *   2022.06.24 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093969/ONNX+Community+Day+-+2022+June+24>
    *   2023.06.28 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14094507/ONNX+Community+Day+2023+-+June+28>

*   **Discuss:**  Open [Issues](https://github.com/onnx/onnx/issues) or use [Slack](https://lfaifoundation.slack.com/) (join [here](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA)).
*   **Follow Us:** Stay updated on ONNX news: [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter](https://twitter.com/onnxai)]

## Roadmap

*   Find the roadmap [here](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

*   [ONNX weekly packages](https://pypi.org/project/onnx-weekly/)
*   Detailed install instructions, including Common Build Options and Common Errors can be found [here](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

*   ONNX uses [pytest](https://docs.pytest.org) as test driver.
*   Install `pytest`:
    ```bash
    pip install pytest
    ```
*   Run tests:
    ```bash
    pytest
    ```

## Development

*   Check out the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for instructions.

## License

*   [Apache License v2.0](LICENSE)

## Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)