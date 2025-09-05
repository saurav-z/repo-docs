<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: Open Neural Network Exchange

**ONNX is an open standard for representing machine learning models, enabling interoperability between various AI frameworks and streamlining the path from research to production.** Learn more and contribute on the [original repository](https://github.com/onnx/onnx).

## Key Features:

*   **Open Ecosystem:** Empowers AI developers to choose the right tools.
*   **Model Interoperability:** Facilitates seamless transfer of models between different frameworks (e.g., PyTorch, TensorFlow, etc.).
*   **Standardized Format:** Provides an open-source format for AI models, including deep learning and traditional ML.
*   **Extensible Computation Graph:** Defines a flexible model structure.
*   **Built-in Operators:** Offers definitions for a wide range of operators and standard data types.
*   **Focus on Inference:** Primarily geared towards supporting model inference (scoring) capabilities.
*   **Widely Supported:** Compatible with numerous frameworks, tools, and hardware.

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

### Programming Utilities for Working with ONNX Graphs

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

## Contribute

ONNX thrives on community contributions. Join us to shape the future of AI!

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Special Interest Groups (SIGs)](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Add a New Operator](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community

*   [Community Meetings Calendar](https://onnx.ai/calendar)
*   [Previous Community Meetup Content](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14091402/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+2020+April+9)
    *   and other meetups from 2020, 2021, 2022, and 2023 (links provided in original README)
*   [Issues](https://github.com/onnx/onnx/issues)
*   [Slack](https://lfaifoundation.slack.com/) (Join:  [https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA))
*   [Facebook](https://www.facebook.com/onnxai/)
*   [Twitter/X](https://twitter.com/onnxai)

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install the ONNX Python package using pip:

```bash
pip install onnx
```
or for optional reference implementation dependencies:

```bash
pip install onnx[reference]
```

*   [ONNX weekly packages](https://pypi.org/project/onnx-weekly/)
*   [Detailed Install Instructions](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

Run tests using pytest after installing it:

```bash
pip install pytest
pytest
```

## Development

*   [Contributor Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)

## Reproducible Builds (Linux)

ONNX provides reproducible builds for Linux, ensuring identical binary outputs from the same source code.

*   [SOURCE_DATE_EPOCH](https://reproducible-builds.org/docs/source-date-epoch/)

## License

*   [Apache License v2.0](LICENSE)

## Trademark

*   [Trademarks](https://trademarks.justia.com/877/25/onnx-87725026.html)
*   [Trademark Usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)