<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Open Neural Network Exchange (ONNX): Unlock AI Model Interoperability

ONNX (Open Neural Network Exchange) is an open-source ecosystem designed to facilitate seamless interoperability between different AI frameworks and tools, enabling developers to build, train, and deploy AI models more efficiently. Learn more about it on its [original repository](https://github.com/onnx/onnx).

## Key Features of ONNX

*   **Open Standard:** ONNX provides an open standard for representing AI models, ensuring compatibility across various frameworks and platforms.
*   **Broad Support:**  ONNX is widely supported by numerous frameworks, tools, and hardware vendors, including popular deep learning libraries.
*   **Interoperability:** Enables easy model transfer and execution across different AI frameworks, allowing developers to leverage the strengths of each tool.
*   **Extensible Computation Graph:** Defines a flexible computation graph model to support diverse model architectures.
*   **Built-in Operators:** Provides a comprehensive set of built-in operators and standard data types to streamline model creation and deployment.
*   **Focus on Inference:** Primarily focuses on supporting inference (scoring) capabilities, optimizing models for efficient deployment.

## Getting Started with ONNX

### Resources
*   [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   [Pre-trained ONNX models](https://github.com/onnx/models)

## Deep Dive into the ONNX Specification

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

## Contributing to ONNX

ONNX is a community-driven project and welcomes contributions from everyone; join the effort by exploring the guidelines and resources.

*   [Contribution guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Community governance model](https://github.com/onnx/onnx/blob/main/community/readme.md)
*   [Special Interest Groups (SIGs)](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Adding New Operators](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

### Community Meetings
*   [Community Meetings Calendar](https://onnx.ai/calendar)
*   [Community Meetup Archives](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14091402/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+2020+April+9)
*   [Community Meetup Archives](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092138/LF+AI+Day+-+ONNX+Community+Workshop+-+2020+October+14)
*   [Community Meetup Archives](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092424/Instructions+for+Event+Hosts+-+LF+AI+Data+Day+-+ONNX+Virtual+Community+Meetup+-+March+2021)
*   [Community Meetup Archives](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093194/LF+AI+Data+Day+ONNX+Community+Virtual+Meetup+-+October+2021)
*   [Community Meetup Archives](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093969/ONNX+Community+Day+-+2022+June+24)
*   [Community Meetup Archives](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14094507/ONNX+Community+Day+2023+-+June+28)

## Connect with the ONNX Community

*   [Issues](https://github.com/onnx/onnx/issues)
*   [Slack](https://lfaifoundation.slack.com/) (Join via [this link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA))
*   [Facebook](https://www.facebook.com/onnxai/)
*   [Twitter/X](https://twitter.com/onnxai)

## Roadmap & Releases
*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install ONNX using pip.

```bash
pip install onnx
```
```bash
pip install onnx[reference] # for optional reference implementation dependencies
```

### Weekly Releases
Experiment with the latest features and improvements through the weekly ONNX packages.
*   [ONNX weekly packages](https://pypi.org/project/onnx-weekly/)

### Installation Details

*   [Detailed install instructions](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

Run tests using pytest.

```bash
pip install pytest
pytest
```

## Development

Explore the Contributor Guide for details on contributing.

*   [Contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)

## License

[Apache License v2.0](LICENSE)

## Trademark

Check out [https://trademarks.justia.com](https://trademarks.justia.com/877/25/onnx-87725026.html) for the trademark.
*   [General rules of the Linux Foundation on Trademark usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)