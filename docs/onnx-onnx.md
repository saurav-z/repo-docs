<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: The Open Standard for AI Models

**ONNX (Open Neural Network Exchange) is an open-source format that simplifies the AI model lifecycle, allowing you to seamlessly move between frameworks and platforms.** Explore the official repository [here](https://github.com/onnx/onnx).

## Key Features

*   **Open Ecosystem:** Empowers AI developers to choose the right tools for their projects.
*   **Model Interoperability:** Enables seamless transitions between diverse AI frameworks, tools, and hardware.
*   **Standard Format:** Provides a universal, open-source format for both deep learning and traditional ML models.
*   **Extensible Computation Graph:** Defines a flexible and adaptable computation graph model.
*   **Built-in Operators:** Includes a comprehensive set of pre-defined operators and standard data types.
*   **Focus on Inference:** Primarily designed to support and optimize model inference (scoring).

## Getting Started

### Use ONNX
*   [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   [Pre-trained ONNX models](https://github.com/onnx/models)

## Dive Deeper into ONNX

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

ONNX is a community-driven project.  Join us!

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Special Interest Groups (SIGs)](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)

For new operator proposals, see [this document](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md).

## Community

*   [Community Meetings Calendar](https://onnx.ai/calendar)
*   [Discuss](https://github.com/onnx/onnx/issues) and [Slack](https://lfaifoundation.slack.com/)
*   Stay connected on [Facebook](https://www.facebook.com/onnxai/) and [Twitter/X](https://twitter.com/onnxai).

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install ONNX via pip:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

Weekly packages are available: [ONNX weekly packages](https://pypi.org/project/onnx-weekly/)

Detailed install instructions [here](https://github.com/onnx/onnx/blob/main/INSTALL.md).

## Testing

Run tests using pytest:

```bash
pip install pytest
pytest
```

## Development

See the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md).

## License

[Apache License v2.0](LICENSE)

## Trademark

[https://trademarks.justia.com/877/25/onnx-87725026.html](https://trademarks.justia.com/877/25/onnx-87725026.html)

[General rules of the Linux Foundation on Trademark usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)