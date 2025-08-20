<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: Open Neural Network Exchange

**ONNX is an open standard for representing machine learning models, enabling interoperability and accelerating AI innovation.**  For the original repository, see [here](https://github.com/onnx/onnx).

## Key Features

*   **Open Source Format:** Provides a standardized format for AI models, encompassing both deep learning and traditional ML models.
*   **Interoperability:** Facilitates the seamless exchange of models between different AI frameworks, tools, and hardware platforms.
*   **Extensible Computation Graph:** Defines a flexible computational graph model.
*   **Built-in Operators & Data Types:** Includes a rich set of built-in operators and standard data type definitions.
*   **Focus on Inferencing:** Primarily designed to support the inference (scoring) phase of AI models.
*   **Community Driven:** Evolved through community contributions, fostering a collaborative ecosystem for AI development.

## Getting Started

*   **Documentation:** [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   **Tutorials:** [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   **Pre-trained Models:** [Pre-trained ONNX models](https://github.com/onnx/models)

## Learn More About the ONNX Specification

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

## Contribute to ONNX

ONNX is a community-driven project, and contributions are welcome.

*   **Contribution Guide:** [Contribution guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   **Special Interest Groups (SIGs):** [SIGs](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   **Working Groups:** [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)

## Community

*   **Community Meetings:** [Meeting Schedules](https://onnx.ai/calendar)
*   **Discuss:** [Issues](https://github.com/onnx/onnx/issues) and [Slack](https://lfaifoundation.slack.com/)
*   **Follow Us:** [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter/X](https://twitter.com/onnxai)]

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install ONNX using pip:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

For detailed installation instructions, see [INSTALL.md](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

Run tests using pytest:

```bash
pip install pytest
pytest
```

## Development

Refer to the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for instructions.

## License

[Apache License v2.0](LICENSE)

## Trademark

[https://trademarks.justia.com/877/25/onnx-87725026.html](https://trademarks.justia.com/877/25/onnx-87725026.html)
[General rules of the Linux Foundation on Trademark usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)