<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

# ONNX: Open Neural Network Exchange

**ONNX (Open Neural Network Exchange) is an open standard for representing machine learning models, enabling interoperability between AI frameworks and streamlining the path from research to production.**

[View the original repository on GitHub](https://github.com/onnx/onnx)

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Features of ONNX:

*   **Open Standard:** Provides an open source format for AI models, promoting interoperability.
*   **Framework Agnostic:** Supports models from various deep learning and machine learning frameworks.
*   **Extensible Model:** Defines an extensible computation graph model.
*   **Built-in Operators:** Includes definitions for a comprehensive set of built-in operators and standard data types.
*   **Focus on Inference:** Currently optimized for model inferencing (scoring) capabilities.
*   **Community Driven:** Actively developed and supported by a vibrant community.

## Getting Started

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

## Contribute

ONNX thrives on community contributions. Join us!

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Special Interest Groups](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Adding New Operators](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community

*   [Community Meetings Calendar](https://onnx.ai/calendar)
*   [Discuss with the Community](https://github.com/onnx/onnx/issues) or on [Slack](https://lfaifoundation.slack.com/) (use this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA) to join)
*   [ONNX Community Day](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14094507/ONNX+Community+Day+2023+-+June+28)

## Stay Connected

*   [Facebook](https://www.facebook.com/onnxai/)
*   [Twitter/X](https://twitter.com/onnxai)

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install the latest ONNX release from PyPI:

```bash
pip install onnx  # or pip install onnx[reference] for optional reference implementation dependencies
```

[ONNX weekly packages](https://pypi.org/project/onnx-weekly/) are available for early access.

For detailed installation instructions, including build options and troubleshooting, see the [INSTALL.md](https://github.com/onnx/onnx/blob/main/INSTALL.md) file.

## Testing

Run the ONNX tests using `pytest`:

1.  Install pytest: `pip install pytest`
2.  Run tests: `pytest`

## Development

Refer to the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for development instructions.

## License

[Apache License v2.0](LICENSE)

## Trademark

*   [Trademark Information](https://trademarks.justia.com/877/25/onnx-87725026.html)
*   [Linux Foundation Trademark Usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)