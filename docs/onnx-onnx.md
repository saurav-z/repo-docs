<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" alt="ONNX Logo" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: Open Neural Network Exchange

**ONNX (Open Neural Network Exchange) is an open standard for representing machine learning models, enabling interoperability between various AI frameworks and tools.**

[Visit the original repository](https://github.com/onnx/onnx) for more details.

## Key Features

*   **Open Format:** Provides a standardized format for AI models, facilitating model exchange between different frameworks.
*   **Framework Agnostic:** Supports a wide range of deep learning and machine learning frameworks, including PyTorch, TensorFlow, and more.
*   **Interoperability:** Enables seamless transfer of models between different platforms and tools, promoting collaboration and innovation.
*   **Extensible Computation Graph:** Defines an adaptable computation graph model, allowing for the representation of complex model architectures.
*   **Standardized Operators:** Includes a set of built-in operators and data types for consistent model interpretation.

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

### Programming utilities for working with ONNX Graphs

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

## Contribute

ONNX is a community-driven project, and contributions are welcome.

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Open Governance Model](https://github.com/onnx/onnx/blob/main/community/readme.md)
*   [Special Interest Groups (SIGs)](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Adding Operators](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community

*   [Community Meetings Calendar](https://onnx.ai/calendar)
*   [Discussion](https://github.com/onnx/onnx/issues)
*   [Slack](https://lfaifoundation.slack.com/) ([Join Slack](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA))
*   [Facebook](https://www.facebook.com/onnxai/)
*   [Twitter](https://twitter.com/onnxai)

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install the ONNX package from PyPI:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

### Weekly Packages

Experiment with the latest features by installing weekly packages:

```bash
pip install onnx-weekly
```

### Detailed Installation Instructions

*   [Installation Guide](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

Run tests using pytest:

1.  Install pytest: `pip install pytest`
2.  Run tests: `pytest`

## Development

*   [Contributor Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)

## License

[Apache License v2.0](LICENSE)

## Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)