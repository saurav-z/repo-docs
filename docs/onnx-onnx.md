<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: Open Neural Network Exchange

**ONNX is an open standard for representing machine learning models, enabling interoperability and streamlining AI workflows.**

## Key Features of ONNX:

*   **Open Format:** Defines a standard format for AI models, supporting both deep learning and traditional ML.
*   **Interoperability:** Facilitates the exchange of models between different frameworks, tools, and hardware platforms.
*   **Extensible Computation Graph:** Provides a flexible model for representing computations.
*   **Operator Definitions:** Includes a set of built-in operators and standard data types.
*   **Focus on Inference:** Primarily designed for efficient model inference (scoring) in production.
*   **Community-Driven:** Encourages community contribution to evolve the standard.

## Getting Started

*   **Documentation:** [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   **Tutorials:** [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   **Pre-trained Models:** [Pre-trained ONNX models](https://github.com/onnx/models)

## Learn More About the ONNX Specification

*   [Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md)
*   [ONNX intermediate representation spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)
*   [Versioning principles of the spec](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
*   [Operators Documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
*   [Operators Documentation (Latest Release)](https://onnx.ai/onnx/operators/index.html)
*   [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)

## Utilities for Working with ONNX Graphs

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

## Contribute to ONNX

ONNX thrives on community contributions.  [Join us on GitHub](https://github.com/onnx/onnx)!

*   **Community Participation:** [Open Governance Model](https://github.com/onnx/onnx/blob/main/community/readme.md)
*   **Special Interest Groups (SIGs):** [SIGs](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   **Working Groups:** [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   **Contribution Guide:** [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   **Adding New Operators:** [Adding New Operators](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community Meetings

Stay informed about the ONNX community:

*   [Meeting Schedules](https://onnx.ai/calendar)
*   [Community Meetup Content](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14091402/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+2020+April+9)
*   [Community Meetup Content](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092138/LF+AI+Day+-+ONNX+Community+Workshop+-+2020+October+14)
*   [Community Meetup Content](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092424/Instructions+for+Event+Hosts+-+LF+AI+Data+Day+-+ONNX+Virtual+Community+Meetup+-+March+2021)
*   [Community Meetup Content](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093194/LF+AI+Data+Day+ONNX+Community+Virtual+Meetup+-+October+2021)
*   [Community Meetup Content](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093969/ONNX+Community+Day+-+2022+June+24)
*   [Community Meetup Content](https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14094507/ONNX+Community+Day+2023+-+June+28)

## Discuss

*   [Issues](https://github.com/onnx/onnx/issues)
*   [Slack](https://lfaifoundation.slack.com/) (Use this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA) to join)

## Stay Connected

*   [Facebook](https://www.facebook.com/onnxai/)
*   [Twitter/X](https://twitter.com/onnxai)

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install the ONNX Python package:

```bash
pip install onnx
```
or install the reference implementation dependencies:
```bash
pip install onnx[reference]
```

*   **Weekly Packages:** [ONNX Weekly Packages](https://pypi.org/project/onnx-weekly/)
*   **Detailed Instructions:** [INSTALL.md](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

Run tests using pytest:

```bash
pip install pytest
pytest
```

## Development

*   [Contributor Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)

## License

*   [Apache License v2.0](LICENSE)

## Trademark

*   [Trademark Information](https://trademarks.justia.com/877/25/onnx-87725026.html)
*   [Linux Foundation Trademark Usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)