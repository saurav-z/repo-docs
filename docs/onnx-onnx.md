<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/onnx.svg)](https://pypi.org/project/onnx)
[![CI](https://github.com/onnx/onnx/actions/workflows/main.yml/badge.svg)](https://github.com/onnx/onnx/actions/workflows/main.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/onnx/onnx/badge)](https://api.securityscorecards.dev/projects/github.com/onnx/onnx)
[![REUSE compliant](https://api.reuse.software/badge/github.com/onnx/onnx)](https://api.reuse.software/info/github.com/onnx/onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ONNX: Open Neural Network Exchange

**ONNX is an open standard that allows AI developers to easily move models between different frameworks and tools.**

ONNX, the Open Neural Network Exchange, is a powerful open-source format for representing machine learning models, including both deep learning and traditional ML, enabling interoperability and streamlining AI development.  For more information, visit the [original repository](https://github.com/onnx/onnx).

## Key Features

*   **Model Portability:** Enables seamless transfer of AI models between various frameworks like PyTorch, TensorFlow, and others.
*   **Open Ecosystem:** Fosters collaboration and innovation with a community-driven approach.
*   **Extensible Computation Graph:** Defines a flexible graph model for representing model computations.
*   **Operator Definitions:** Includes built-in operators and standard data types for efficient model execution.
*   **Focus on Inference:** Primarily designed for optimizing and deploying AI models for inference (scoring) tasks.

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

ONNX thrives on community contributions.  Join us!

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Special Interest Groups](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)

### Adding Operators
*   [Adding New Operators](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community

*   **Community Meetings:** Find schedules [here](https://onnx.ai/calendar)
*   **Community Meetups:**
    *   2020.04.09 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14091402/LF+AI+Day+-ONNX+Community+Virtual+Meetup+-+Silicon+Valley+-+2020+April+9>
    *   2020.10.14 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092138/LF+AI+Day+-+ONNX+Community+Workshop+-+2020+October+14>
    *   2021.03.24 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14092424/Instructions+for+Event+Hosts+-+LF+AI+Data+Day+-+ONNX+Virtual+Community+Meetup+-+March+2021>
    *   2021.10.21 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093194/LF+AI+Data+Day+ONNX+Community+Virtual+Meetup+-+October+2021>
    *   2022.06.24 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14093969/ONNX+Community+Day+-+2022+June+24>
    *   2023.06.28 <https://lf-aidata.atlassian.net/wiki/spaces/DL/pages/14094507/ONNX+Community+Day+2023+-+June+28>
*   **Discuss:** Engage in discussions via [Issues](https://github.com/onnx/onnx/issues) and [Slack](https://lfaifoundation.slack.com/) (join [here](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA)).
*   **Follow Us:** Stay updated on social media: [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter/X](https://twitter.com/onnxai)]

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install ONNX using pip:

```bash
pip install onnx
```

For optional reference implementation dependencies:

```bash
pip install onnx[reference]
```

### Weekly Packages
*   [ONNX Weekly Packages](https://pypi.org/project/onnx-weekly/)

### Detailed Instructions
*   [Installation Instructions](https://github.com/onnx/onnx/blob/main/INSTALL.md)

## Testing

Run tests with pytest:

1.  Install pytest:

    ```bash
    pip install pytest
    ```
2.  Execute tests:

    ```bash
    pytest
    ```

## Development

*   [Contributor Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)

## License

*   [Apache License v2.0](LICENSE)

## Trademark

*   [ONNX Trademark](https://trademarks.justia.com/877/25/onnx-87725026.html)
*   [Linux Foundation Trademark Usage](https://www.linuxfoundation.org/legal/trademark-usage)

## Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)