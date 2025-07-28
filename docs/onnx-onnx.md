<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" alt="ONNX Logo"/></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ONNX: Open Neural Network Exchange

**ONNX is the open standard for representing machine learning models, enabling seamless interoperability between diverse AI frameworks and tools.**

### Key Features:

*   **Open Ecosystem:** ONNX facilitates the use of the right tools for your AI project by providing an open-source format.
*   **Model Interoperability:** Supports both deep learning and traditional ML models, enabling the exchange of models between different frameworks.
*   **Extensible Computation Graph:** Defines a flexible computation graph model, along with built-in operators and standard data types.
*   **Widely Supported:**  ONNX is supported by a broad range of frameworks, tools, and hardware, increasing the speed of innovation.
*   **Focus on Inference:** Primarily designed for inference (scoring) capabilities.

For more information, visit the [original repository](https://github.com/onnx/onnx).

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

### Programming Utilities for ONNX Graphs

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

### Contribute

ONNX is a community-driven project. Join us to contribute!

*   [Community Governance](https://github.com/onnx/onnx/blob/main/community/readme.md)
*   [Special Interest Groups](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Add a New Operator](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

### Community

*   [Community meetings calendar](https://onnx.ai/calendar)
*   [Discuss](https://github.com/onnx/onnx/issues) or [Slack](https://lfaifoundation.slack.com/)
*   [Join Slack](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA)
*   [Follow Us on Facebook](https://www.facebook.com/onnxai/)
*   [Follow Us on Twitter](https://twitter.com/onnxai)

### Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

### Installation

Install ONNX using pip:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

*   [ONNX Weekly Packages](https://pypi.org/project/onnx-weekly/)
*   [Installation Instructions](https://github.com/onnx/onnx/blob/main/INSTALL.md)

### Testing

Run tests using pytest:

```bash
pip install pytest
pytest
```

### Development

*   [Contributor Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)

### License

*   [Apache License v2.0](LICENSE)

### Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)