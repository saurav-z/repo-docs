<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/main/docs/onnx-horizontal-color.png" /></p>

# ONNX: Open Neural Network Exchange

**ONNX empowers AI developers with a universal format for AI models, fostering interoperability and accelerating innovation in the AI community.**

[ONNX](https://github.com/onnx/onnx) is an open-source ecosystem that provides a standard format for representing machine learning models. This allows for seamless transitions between different AI frameworks, tools, and hardware platforms.

## Key Features:

*   **Open Standard:** Defines a common format for AI models, enabling interoperability.
*   **Broad Support:**  Widely supported across numerous frameworks, tools, and hardware platforms.
*   **Extensible Computation Graph:** Provides a flexible and adaptable model for deep learning and traditional ML.
*   **Built-in Operators:** Includes a comprehensive set of pre-defined operators for common AI tasks.
*   **Standard Data Types:**  Uses standard data types for consistency and compatibility.
*   **Focus on Inference:** Primarily designed to optimize the process of inferencing and scoring AI models.

## Getting Started

*   **Documentation:** [Documentation of ONNX Python Package](https://onnx.ai/onnx/)
*   **Tutorials:** [Tutorials for creating ONNX models](https://github.com/onnx/tutorials)
*   **Pre-trained Models:** [Pre-trained ONNX models](https://github.com/onnx/models)

## Learn More

*   **Overview:** [Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md)
*   **Intermediate Representation Spec:** [ONNX intermediate representation spec](https://github.com/onnx/onnx/blob/main/docs/IR.md)
*   **Versioning Principles:** [Versioning principles of the spec](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
*   **Operators Documentation:** [Operators documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md) and [Operators documentation](https://onnx.ai/onnx/operators/index.html)
*   **Python API Overview:** [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)

## Programming Utilities

*   **Shape and Type Inference:** [Shape and Type Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
*   **Graph Optimization:** [Graph Optimization](https://github.com/onnx/optimizer)
*   **Opset Version Conversion:** [Opset Version Conversion](https://github.com/onnx/onnx/blob/main/docs/docsgen/source/api/version_converter.md)

## Contribute

ONNX thrives on community contributions. We welcome your participation in shaping the future of ONNX!

*   **Contribution Guide:** [Contribution guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md)
*   **Special Interest Groups (SIGs):** [Special Interest Groups](https://github.com/onnx/onnx/blob/main/community/sigs.md)
*   **Working Groups:** [Working Groups](https://github.com/onnx/onnx/blob/main/community/working-groups.md)
*   **Adding Operators:** [Adding New Operators](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md)

## Community

*   **Community Meetings:** Schedules are found [here](https://onnx.ai/calendar)
*   **Community Meetups:** Find content from past meetups at the links provided in the original README.
*   **Discussion:** [Issues](https://github.com/onnx/onnx/issues) and [Slack](https://lfaifoundation.slack.com/) (join [here](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA))
*   **Stay Connected:** [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter](https://twitter.com/onnxai)]
*   **Roadmap:** Find it [here](https://github.com/onnx/steering-committee/tree/main/roadmap)

## Installation

Install ONNX from PyPI:

```bash
pip install onnx # or pip install onnx[reference] for optional reference implementation dependencies
```

For weekly packages:

```bash
pip install onnx-weekly
```

See [INSTALL.md](https://github.com/onnx/onnx/blob/main/INSTALL.md) for detailed installation instructions, build options, and troubleshooting.

## Testing

ONNX uses `pytest` for testing.

1.  **Install pytest:**

    ```bash
    pip install pytest
    ```

2.  **Run tests:**

    ```bash
    pytest
    ```

## Development

See the [contributor guide](https://github.com/onnx/onnx/blob/main/CONTRIBUTING.md) for development instructions.

## License

[Apache License v2.0](LICENSE)

## Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)