<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: Open Neural Network Exchange

**ONNX is an open ecosystem that empowers AI developers with a universal format for AI models, enabling seamless interoperability and accelerating innovation.**

[View the original repository on GitHub](https://github.com/onnx/onnx)

## Key Features of ONNX:

*   **Open Standard:** Defines a standard format for AI models, facilitating interoperability between different frameworks.
*   **Framework Agnostic:** Supports a wide range of deep learning and traditional machine learning frameworks.
*   **Interoperability:** Enables easy movement of models between different tools and hardware.
*   **Extensible:** Features a flexible and extensible computation graph model, accommodating evolving AI needs.
*   **Community-Driven:** A collaborative project with active community participation, fostering continuous improvement.

## Getting Started with ONNX

### Use ONNX Resources

*   [ONNX Python Package Documentation](https://onnx.ai/onnx/)
*   [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   [Pre-trained ONNX models](https://github.com/onnx/models)

### Learn About the ONNX Specification

*   [Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md)
*   [ONNX Intermediate Representation (IR) Spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)
*   [Versioning Principles of the Spec](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
*   [Operators Documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
*   [Operators Documentation](https://onnx.ai/onnx/operators/index.html) (latest release)
*   [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)

### Programming Utilities

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

## Contribute to ONNX

ONNX thrives on community contributions.  Learn how to get involved:

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Special Interest Groups (SIGs)](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)

## Community

### Community Meetings

Find schedules for regular meetings of the Steering Committee, working groups, and SIGs [here](https://onnx.ai/calendar).

### Community Meetups

Access content from past community meetups:

*   2020.04.09 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14091402/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+2020+April+9>
*   2020.10.14 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092138/LF+AI+Day+-+ONNX+Community+Workshop+-+2020+October+14>
*   2021.03.24 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092424/Instructions+for+Event+Hosts+-+LF+AI+Data+Day+-+ONNX+Virtual+Community+Meetup+-+March+2021>
*   2021.10.21 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093194/LF+AI+Data+Day+ONNX+Community+Virtual+Meetup+-+October+2021>
*   2022.06.24 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093969/ONNX+Community+Day+-+2022+June+24>
*   2023.06.28 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14094507/ONNX+Community+Day+2023+-+June+28>

### Discuss

*   [Issues](https://github.com/onnx/onnx/issues)
*   [Slack](https://lfaifoundation.slack.com/) (If you have not joined yet, please use this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA) to join the group) for more real-time discussion.

### Follow Us

Stay up-to-date with the latest ONNX news:  [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter/X](https://twitter.com/onnxai)]

## Roadmap

Find details about the annual roadmap process [here](https://github.com/onnx/steering-committee/tree/main/roadmap).

## Installation

ONNX packages are available on PyPI.

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

[ONNX weekly packages](https://pypi.org/project/onnx-weekly/) are published in PyPI to enable experimentation and early testing.

For detailed installation instructions including common build options and errors, refer to the [INSTALL.md](https://github.com/onnx/onnx/blob/main/INSTALL.md) file.

## Testing

ONNX uses [pytest](https://docs.pytest.org) for testing.  Install pytest and run tests as follows:

```bash
pip install pytest
pytest
```

## Development

See the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for development instructions.

## License

[Apache License v2.0](LICENSE)

## Trademark

Checkout [https://trademarks.justia.com](https://trademarks.justia.com/877/25/onnx-87725026.html) for the trademark.

[General rules of the Linux Foundation on Trademark usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)