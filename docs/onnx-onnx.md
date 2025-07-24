<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

# ONNX: Open Neural Network Exchange

**ONNX is an open standard for representing machine learning models, enabling interoperability and accelerating AI innovation.**

[Original Repository](https://github.com/onnx/onnx)

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Features

*   **Open Standard:**  Defines a common format for AI models, including deep learning and traditional ML, promoting interoperability.
*   **Broad Framework Support:**  Widely supported by numerous frameworks, tools, and hardware platforms, facilitating model portability.
*   **Extensible Computation Graph:** Utilizes a flexible computation graph model that allows for the representation of complex AI models.
*   **Built-in Operators & Data Types:** Provides a rich set of built-in operators and standard data types, ensuring model consistency.
*   **Focus on Inference:** Currently optimized for inferencing (scoring) capabilities, enabling efficient model deployment.

## Getting Started

*   [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   [Pre-trained ONNX models](https://github.com/onnx/models)

## Learn the ONNX Specification

*   [Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md)
*   [ONNX intermediate representation spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)
*   [Versioning principles of the spec](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
*   [Operators documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
*   [Operators documentation](https://onnx.ai/onnx/operators/index.html) (latest release)
*   [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)

## Programming Utilities

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

## Contribute

ONNX thrives on community contributions. Learn how you can get involved:

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Community Governance](https://github.com/onnx/onnx/blob/main/community/readme.md)
*   [Special Interest Groups (SIGs)](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Adding New Operators](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community

*   [Community Meetings Calendar](https://onnx.ai/calendar)
*   [Discussion](https://github.com/onnx/onnx/issues)
*   [Slack](https://lfaifoundation.slack.com/) (If you have not joined yet, please use this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA) to join the group)

## Stay Connected

*   [Facebook](https://www.facebook.com/onnxai/)
*   [Twitter](https://twitter.com/onnxai)

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install ONNX using pip:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

[ONNX weekly packages](https://pypi.org/project/onnx-weekly/) are published in PyPI to enable experimentation and early testing.
For detailed installation instructions, see [INSTALL.md](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

Run tests using pytest:

```bash
pip install pytest
pytest
```

## Development

Refer to the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for development instructions.

## License

[Apache License v2.0](LICENSE)

## Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)