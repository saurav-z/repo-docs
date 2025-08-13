<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: Open Neural Network Exchange

**ONNX (Open Neural Network Exchange) is an open ecosystem empowering AI developers with a flexible format for AI models, promoting interoperability and accelerating innovation.** This README provides a comprehensive overview. You can find the original repository [here](https://github.com/onnx/onnx).

## Key Features

*   **Open Standard:** A universal format for representing machine learning models, fostering interoperability between various frameworks and tools.
*   **Framework Support:**  Widely supported across various deep learning frameworks, including TensorFlow, PyTorch, and more.  [See supported tools](http://onnx.ai/supported-tools).
*   **Model Interoperability:** Enables seamless conversion and deployment of models across different platforms and hardware.
*   **Extensible Computation Graph:** Defines a flexible and extensible computation graph model.
*   **Built-in Operators & Data Types:** Includes definitions for a range of built-in operators and standard data types.
*   **Focus on Inference (Scoring):** Primarily focused on capabilities needed for efficient model inference and deployment.

## Getting Started

### Use ONNX

*   [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   [Pre-trained ONNX models](https://github.com/onnx/models)

### Learn about the ONNX spec

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

ONNX is a community-driven project.  Your contributions are welcome!

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Special Interest Groups (SIGs)](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Adding a new operator](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community

*   [Community Meetings Calendar](https://onnx.ai/calendar)
*   [Discuss](https://github.com/onnx/onnx/issues) and [Slack](https://lfaifoundation.slack.com/)
*   Stay up to date with the latest ONNX news on [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter/X](https://twitter.com/onnxai)]

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install the latest ONNX package using pip:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

For more information:

*   [ONNX weekly packages](https://pypi.org/project/onnx-weekly/)
*   [Detailed install instructions](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

Run tests using pytest:

1.  Install pytest: `pip install pytest`
2.  Run tests: `pytest`

## Development

*   Check out the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for instructions.

## License

*   [Apache License v2.0](LICENSE)

## Trademark

*   Checkout [https://trademarks.justia.com/877/25/onnx-87725026.html](https://trademarks.justia.com/877/25/onnx-87725026.html) for the trademark.
*   [General rules of the Linux Foundation on Trademark usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)