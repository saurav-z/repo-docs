<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ONNX: Open Neural Network Exchange

**ONNX is the open standard for representing machine learning models, enabling interoperability and simplifying the AI development lifecycle.** This project provides the core components of the ONNX ecosystem.  For more information, visit the [original repository](https://github.com/onnx/onnx).

### Key Features:

*   **Open Format:** Defines a standard format for AI models, encompassing both deep learning and traditional machine learning.
*   **Interoperability:** Enables seamless model exchange between different frameworks, tools, and hardware platforms.
*   **Extensible Computation Graph:**  Provides a flexible model for representing computations.
*   **Built-in Operators:** Includes a comprehensive set of predefined operators for common AI tasks.
*   **Standard Data Types:** Utilizes standard data types to ensure consistency across different platforms.

### Getting Started

*   [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   [Pre-trained ONNX models](https://github.com/onnx/models)

### Learn About the ONNX Specification

*   [Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md)
*   [ONNX intermediate representation spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)
*   [Versioning principles of the spec](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
*   [Operators documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
*   [Operators documentation](https://onnx.ai/onnx/operators/index.html) (latest release)
*   [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)

### Programming Utilities

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

### Contribute

ONNX is a community-driven project.  Your contributions are welcome!

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Special Interest Groups](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Add New Operator](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

### Community

*   [Community Meetings](https://onnx.ai/calendar)
*   [Discuss: Issues](https://github.com/onnx/onnx/issues) and [Slack](https://lfaifoundation.slack.com/)
*   [Follow Us: Facebook](https://www.facebook.com/onnxai/) and [Twitter](https://twitter.com/onnxai)
*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

### Installation

Install ONNX using pip:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

*   [ONNX weekly packages](https://pypi.org/project/onnx-weekly/)

Detailed installation instructions are available in the [INSTALL.md](https://github.com/onnx/onnx/blob/main/INSTALL.md) file.

### Testing

Run tests using pytest:

1.  Install pytest: `pip install pytest`
2.  Run tests: `pytest`

### Development

Review the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for development instructions.

### License

[Apache License v2.0](LICENSE)

### Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)