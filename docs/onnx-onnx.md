<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: Open Neural Network Exchange

**ONNX enables AI developers to easily move between frameworks and streamline the journey from research to production.** This project, hosted on [GitHub](https://github.com/onnx/onnx), provides a standard format for AI models, promoting interoperability and accelerating innovation in the AI community.

## Key Features

*   **Open Standard:** Defines an open format for AI models, supporting both deep learning and traditional ML.
*   **Extensible Computation Graph Model:** Provides a flexible and adaptable structure for representing AI models.
*   **Built-in Operators & Data Types:** Offers a comprehensive set of pre-defined operators and standard data types.
*   **Framework Interoperability:** Facilitates the seamless exchange of models between different AI frameworks.
*   **Focus on Inference:** Primarily supports model inferencing (scoring) capabilities.
*   **Community-Driven:**  An open-source project that encourages community contributions and collaboration.

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

### Programming utilities for working with ONNX Graphs

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

## Contribute

ONNX thrives on community contributions!  Learn how to get involved:

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Community Governance](https://github.com/onnx/onnx/blob/main/community/readme.md)
*   [Special Interest Groups (SIGs)](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Adding New Operators](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

### Community Meetings
Stay connected and informed.

*   [Community Meetings Calendar](https://onnx.ai/calendar)
*   [Past Community Meetups](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14091402/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+2020+April+9)
*   [Past Community Meetups](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092138/LF+AI+Day+-+ONNX+Community+Workshop+-+2020+October+14)
*   [Past Community Meetups](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092424/Instructions+for+Event+Hosts+-+LF+AI+Data+Day+-+ONNX+Virtual+Community+Meetup+-+March+2021)
*   [Past Community Meetups](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093194/LF+AI+Data+Day+ONNX+Community+Virtual+Meetup+-+October+2021)
*   [Past Community Meetups](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093969/ONNX+Community+Day+-+2022+June+24)
*   [Past Community Meetups](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14094507/ONNX+Community+Day+2023+-+June+28)

## Discuss & Stay Connected

*   [Issues](https://github.com/onnx/onnx/issues)
*   [Slack](https://lfaifoundation.slack.com/) (Use this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA) to join the group)
*   [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter/X](https://twitter.com/onnxai)]

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install ONNX easily using pip:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

For experimental releases, try:
*   [ONNX weekly packages](https://pypi.org/project/onnx-weekly/)

For detailed installation instructions and troubleshooting:
*   [Installation Guide](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

Run tests using pytest:

1.  **Install pytest:** `pip install pytest`
2.  **Run tests:** `pytest`

## Development

Explore our [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for development instructions.

## License

[Apache License v2.0](LICENSE)

## Trademark

*   [Trademark Information](https://trademarks.justia.com/877/25/onnx-87725026.html)
*   [Linux Foundation Trademark Usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)