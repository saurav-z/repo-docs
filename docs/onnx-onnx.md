<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: Open Neural Network Exchange

**ONNX (Open Neural Network Exchange) is the open standard for representing machine learning models, enabling interoperability between different AI frameworks and streamlining the path from research to production.**  Learn more and contribute on the [original repo](https://github.com/onnx/onnx).

## Key Features

*   **Open Ecosystem:** Provides an open-source format for AI models, supporting both deep learning and traditional ML.
*   **Framework Interoperability:** Enables seamless model exchange between various AI frameworks, tools, and hardware platforms.
*   **Standardized Model Representation:** Defines a common computational graph model, built-in operators, and standard data types.
*   **Focus on Inferencing:** Primarily optimized for model inferencing (scoring) capabilities.
*   **Community-Driven:** Evolved through community contributions, promoting rapid innovation in the AI space.

## Getting Started

*   **Documentation:** [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   **Tutorials:** [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   **Pre-trained Models:** [Pre-trained ONNX models](https://github.com/onnx/models)

## Learn the ONNX Specification

*   [Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md)
*   [ONNX Intermediate Representation Spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)
*   [Versioning Principles](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
*   [Operators Documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
*   [Latest Operators Documentation](https://onnx.ai/onnx/operators/index.html)
*   [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)

## Utilities

*   [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   [Graph Optimization](https://github.com/onnx/optimizer)
*   [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

## Contribute

Join the ONNX community!  We welcome your contributions.

*   **Contribution Guide:** [Contribution guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   **Special Interest Groups (SIGs):** [SIGs](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   **Working Groups:** [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   **Add New Operator:** [Add New Op](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community Meetings

Find meeting schedules here: [Calendar](https://onnx.ai/calendar)

*   **Community Meetup Archives:**
    *   2020.04.09 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14091402/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+2020+April+9>
    *   2020.10.14 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092138/LF+AI+Day+-+ONNX+Community+Workshop+-+2020+October+14>
    *   2021.03.24 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092424/Instructions+for+Event+Hosts+-+LF+AI+Data+Day+-+ONNX+Virtual+Community+Meetup+-+March+2021>
    *   2021.10.21 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093194/LF+AI+Data+Day+ONNX+Community+Virtual+Meetup+-+October+2021>
    *   2022.06.24 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093969/ONNX+Community+Day+-+2022+June+24>
    *   2023.06.28 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14094507/ONNX+Community+Day+2023+-+June+28>

## Discuss

*   **Issues:** [Issues](https://github.com/onnx/onnx/issues)
*   **Slack:** [Slack](https://lfaifoundation.slack.com/) (Use this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA) to join)

## Follow Us

*   **Facebook:** [[Facebook](https://www.facebook.com/onnxai/)]
*   **Twitter/X:** [[Twitter/X](https://twitter.com/onnxai)]

## Roadmap

*   **Roadmap:** [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```
```bash
# ONNX weekly packages
pip install onnx-weekly
```

*   **Detailed Installation:** [Installation](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

```bash
pip install pytest
pytest
```

## Development

*   **Contributor Guide:** [Contributor Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)

## License

[Apache License v2.0](LICENSE)

## Trademark

*   **Trademark:** [https://trademarks.justia.com/877/25/onnx-87725026.html](https://trademarks.justia.com/877/25/onnx-87725026.html)
*   **Trademark Usage Rules:** [General rules of the Linux Foundation on Trademark usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)