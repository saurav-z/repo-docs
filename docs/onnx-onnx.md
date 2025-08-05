<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Open Neural Network Exchange (ONNX): The Open Standard for AI Models

**ONNX empowers AI developers to seamlessly choose the right tools and accelerate their projects with its open ecosystem for AI models.**  This project provides an open source format for AI models, supporting both deep learning and traditional machine learning, along with definitions of built-in operators and standard data types.

**[View the original repository on GitHub](https://github.com/onnx/onnx)**

## Key Features of ONNX:

*   **Open Standard:**  An open format for AI models, fostering interoperability between diverse AI frameworks and tools.
*   **Model Interoperability:** Enables the smooth transfer of models between different frameworks, allowing you to use the best tool for each task.
*   **Extensible Computation Graph Model:** Defines a flexible computation graph model to accommodate various model architectures.
*   **Built-in Operators:** Provides a comprehensive set of built-in operators for common AI tasks.
*   **Standard Data Types:**  Supports standardized data types for consistency and compatibility across different platforms.
*   **Focus on Inferencing:** Primarily focused on the capabilities needed for efficient model inference and deployment.
*   **Community-Driven:**  Evolves through active community contributions and participation.

## Getting Started with ONNX

*   [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   [Pre-trained ONNX models](https://github.com/onnx/models)

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

ONNX is a community project. We encourage contributions and participation!

*   [Community Governance](https://github.com/onnx/onnx/blob/main/community/readme.md)
*   [Special Interest Groups](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Adding new Operators](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community

*   [Community Meetings Calendar](https://onnx.ai/calendar)
*   [Past Community Meetups](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14091402/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+2020+April+9)
*   [Discuss](https://github.com/onnx/onnx/issues)
*   [Slack](https://lfaifoundation.slack.com/)

## Stay Connected

*   [[Facebook](https://www.facebook.com/onnxai/)]
*   [[Twitter/X](https://twitter.com/onnxai)]

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

```sh
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

*   [ONNX weekly packages](https://pypi.org/project/onnx-weekly/)
*   [Detailed install instructions](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

```sh
pip install pytest
pytest
```

## Development

*   [Contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)

## License

[Apache License v2.0](LICENSE)

## Trademark

*   [Trademark](https://trademarks.justia.com/877/25/onnx-87725026.html)
*   [Linux Foundation on Trademark usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)