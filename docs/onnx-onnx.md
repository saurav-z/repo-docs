<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ONNX: The Open Standard for AI Models

**ONNX (Open Neural Network Exchange) is an open ecosystem that simplifies AI development by enabling interoperability between different AI frameworks and streamlining the path from research to production.**

### Key Features

*   **Open Source Format:** ONNX provides an open standard format for AI models, supporting both deep learning and traditional machine learning models.
*   **Extensible Computation Graph:** Defines a flexible computation graph model for representing AI models.
*   **Built-in Operators & Data Types:** Includes a comprehensive set of built-in operators and standard data types for AI model representation.
*   **Wide Support:**  Extensively supported by frameworks, tools, and hardware, fostering interoperability.
*   **Community Driven:** Actively developed and supported by a vibrant community, encouraging contributions and collaboration.

### Getting Started

*   **Documentation:** Comprehensive documentation for the [ONNX Python Package](https://onnx.ai/onnx/) and the [ONNX specification](https://github.com/onnx/onnx).
*   **Tutorials:** Learn how to create ONNX models using the [ONNX Tutorials](https://github.com/onnx/tutorials).
*   **Pre-trained Models:** Access a collection of pre-trained ONNX models on the [ONNX Models](https://github.com/onnx/models) repository.

### Understanding the ONNX Specification

*   [Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md)
*   [ONNX Intermediate Representation (IR) Spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)
*   [Versioning Principles](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
*   [Operators Documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md) &  [latest release](https://onnx.ai/onnx/operators/index.html)
*   [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)

### Programming Utilities

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

### Contributing

Join the ONNX community and contribute to the project!  Learn more about our [open governance model](https://github.com/onnx/onnx/blob/main/community/readme.md) and how to [contribute code, ideas, and feedback](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md).

### Community

*   Participate in [Special Interest Groups (SIGs)](https://github.com/onnx/onnx/blob/main/community/sigs.md) and [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md).
*   Find the schedules of the regular meetings of the Steering Committee, the working groups and the SIGs [here](https://onnx.ai/calendar)
*   Discuss ONNX on [Issues](https://github.com/onnx/onnx/issues) or [Slack](https://lfaifoundation.slack.com/) (use this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA) to join).

### Follow Us

Stay up-to-date with the latest ONNX news on  [[Facebook](https://www.facebook.com/onnxai/)] and [[Twitter/X](https://twitter.com/onnxai)].

### Roadmap

Review the current [roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap) to understand future development plans.

### Installation

Install the ONNX Python package easily using pip:

```bash
pip install onnx
```

or, for reference implementations:

```bash
pip install onnx[reference]
```

For detailed installation instructions, including common build options and error resolution, see [INSTALL.md](https://github.com/onnx/onnx/blob/main/INSTALL.md).

### Testing

ONNX utilizes [pytest](https://docs.pytest.org) for testing.  To run tests, install pytest and execute:

```bash
pip install pytest
pytest
```

### Development

Refer to the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for development instructions.

### License

Licensed under the [Apache License v2.0](LICENSE).

### Trademark

See [https://trademarks.justia.com/877/25/onnx-87725026.html](https://trademarks.justia.com/877/25/onnx-87725026.html) for trademark information.

### Code of Conduct

Adheres to the [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html).

[Back to the top](https://github.com/onnx/onnx)