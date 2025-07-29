<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: Open Neural Network Exchange

**ONNX is an open standard for representing machine learning models, enabling interoperability between various AI frameworks and streamlining the path from research to production.** Learn more and contribute at the [ONNX GitHub repository](https://github.com/onnx/onnx).

## Key Features

*   **Open Format:** Provides a universal format for AI models, supporting both deep learning and traditional ML models.
*   **Extensible Computation Graph:** Defines a flexible computation graph model for representing model structure.
*   **Built-in Operators:** Includes a comprehensive set of predefined operators for common AI tasks.
*   **Standard Data Types:** Uses standard data types to ensure consistency and compatibility across different platforms.
*   **Wide Support:** Supported by numerous frameworks, tools, and hardware platforms.
*   **Interoperability:** Enables seamless model transfer between different AI frameworks, promoting collaboration and innovation.
*   **Focus on Inferencing:** Primarily focuses on the capabilities needed for inferencing (scoring), optimizing for production use.

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

ONNX is a community-driven project; we encourage your contributions and feedback.

*   **Contribution Guide:** [Contribution guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   **Community Governance:** [Community Governance](https://github.com/onnx/onnx/blob/main/community/readme.md)
*   **Special Interest Groups (SIGs):** [SIGs](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   **Working Groups:** [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   **Add New Operators:** [Add New Op](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

### Community Meetings

Stay informed with regular meetings to discuss the project's progress and future directions.

*   Find the schedules of the regular meetings of the Steering Committee, the working groups and the SIGs [here](https://onnx.ai/calendar)

### Community Meetups

*   2020.04.09 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14091402/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+2020+April+9>
*   2020.10.14 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092138/LF+AI+Day+-+ONNX+Community+Workshop+-+2020+October+14>
*   2021.03.24 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092424/Instructions+for+Event+Hosts+-+LF+AI+Data+Day+-+ONNX+Virtual+Community+Meetup+-+March+2021>
*   2021.10.21 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093194/LF+AI+Data+Day+ONNX+Community+Virtual+Meetup+-+October+2021>
*   2022.06.24 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093969/ONNX+Community+Day+-+2022+June+24>
*   2023.06.28 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14094507/ONNX+Community+Day+2023+-+June+28>

## Discuss

*   **Issues:** Open [Issues](https://github.com/onnx/onnx/issues)
*   **Slack:** [Slack](https://lfaifoundation.slack.com/) (Join using this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA))

## Follow Us

*   **Facebook:** [[Facebook](https://www.facebook.com/onnxai/)]
*   **Twitter:** [[Twitter](https://twitter.com/onnxai)]

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install ONNX using pip:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

*   **Weekly Packages:** [ONNX weekly packages](https://pypi.org/project/onnx-weekly/) are available for experimentation and early testing.
*   **Detailed Installation Instructions:** Find detailed install instructions, including Common Build Options and Common Errors [here](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

ONNX uses pytest for testing.

1.  **Install pytest:**

    ```bash
    pip install pytest
    ```

2.  **Run tests:**

    ```bash
    pytest
    ```

## Development

*   Check out the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for instructions.

## License

*   [Apache License v2.0](LICENSE)

## Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)