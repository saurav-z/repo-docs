<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" alt="ONNX Logo"/></p>

# ONNX: Open Neural Network Exchange

**ONNX is an open standard for representing machine learning models, enabling interoperability between different AI frameworks and streamlining the path from research to production.**  

[See the original repository on GitHub](https://github.com/onnx/onnx).

ONNX (Open Neural Network Exchange) is a powerful open ecosystem designed to empower AI developers by providing a standard format for AI models, both deep learning and traditional ML. This enables seamless model exchange and deployment across various frameworks, tools, and hardware platforms.

## Key Features

*   **Open Format:**  Defines a universal format for AI models, enabling interoperability.
*   **Extensible Computation Graph:** Provides a flexible and scalable model representation.
*   **Built-in Operators & Data Types:** Includes a comprehensive set of operators and standard data types.
*   **Wide Support:** Supported by numerous frameworks, tools, and hardware vendors.
*   **Focus on Inference:** Primarily designed for efficient model inference (scoring).
*   **Community-Driven:** Open-source project with active community contributions.

## Getting Started

*   [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   [Pre-trained ONNX models](https://github.com/onnx/models)

## Learn About the ONNX Specification

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

ONNX is a community-driven project, and we encourage your participation!  Learn how to contribute:

*   [Contribution Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   [Community Governance](https://github.com/onnx/onnx/blob/main/community/readme.md)
*   [Special Interest Groups (SIGs)](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   [Adding New Operators](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community

*   [Community Calendar](https://onnx.ai/calendar)
*   [Discuss on Issues](https://github.com/onnx/onnx/issues)
*   [Slack](https://lfaifoundation.slack.com/) (Join via [this link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA))
*   **Follow Us:** [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter](https://twitter.com/onnxai)]

## Roadmap

*   [Roadmap](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install the latest ONNX release using pip:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

For more details, including common build options and error resolutions:
*   [Detailed Installation Instructions](https://github.com/onnx/onnx/blob/main/INSTALL.md)
*   [ONNX weekly packages](https://pypi.org/project/onnx-weekly/)

## Testing

ONNX uses pytest for testing:

1.  **Install pytest:**

    ```bash
    pip install pytest
    ```
2.  **Run tests:**

    ```bash
    pytest
    ```

## Development

*   [Contributor Guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)

## License

*   [Apache License v2.0](LICENSE)

## Code of Conduct

*   [ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)