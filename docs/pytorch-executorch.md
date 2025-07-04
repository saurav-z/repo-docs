<div align="center">
  <img src="docs/source/_static/img/et-logo.png" alt="ExecuTorch Logo" width="200">
  <h1>ExecuTorch: The Leading On-Device AI Framework for Performance and Portability</h1>
</div>

<div align="center">
  <a href="https://github.com/pytorch/executorch/graphs/contributors"><img src="https://img.shields.io/github/contributors/pytorch/executorch?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/pytorch/executorch/stargazers"><img src="https://img.shields.io/github/stars/pytorch/executorch?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://discord.gg/Dh43CKSAdc"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://pytorch.org/executorch/main/index"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <hr>
</div>

ExecuTorch is a powerful, open-source AI framework from PyTorch, designed for high-performance on-device inference and training across a wide range of platforms.  [Learn more on GitHub](https://github.com/pytorch/executorch).

## Key Features and Benefits

ExecuTorch offers a comprehensive solution for deploying and running AI models directly on devices, with key advantages including:

*   **Broad Platform Support:**  Runs on iOS, macOS (ARM64), Android, Linux, and even microcontrollers, enabling wide deployment.
*   **Hardware Acceleration:** Optimizes performance by leveraging various hardware accelerators like Apple, Arm, Cadence, MediaTek, OpenVINO, Qualcomm, Vulkan, and XNNPACK.
*   **Portability:**  Seamlessly deploy models from high-end mobile devices to resource-constrained embedded systems.
*   **Productivity:** Leverages existing PyTorch tools and workflows for model authoring, conversion, debugging, and deployment, streamlining development.
*   **Performance:**  Provides a lightweight runtime that maximizes hardware capabilities (CPUs, NPUs, DSPs) for a smooth user experience.
*   **Extensive Model Support:**  Compatible with a variety of model types including Large Language Models (LLMs), Computer Vision (CV), Automatic Speech Recognition (ASR), and Text-to-Speech (TTS) models.

## Getting Started

Ready to explore ExecuTorch? Get started with these resources:

*   **Step-by-Step Tutorial:** [Getting Started](https://pytorch.org/executorch/stable/getting-started.html) to get things running locally and deploy a model to a device.
*   **Interactive Colab Notebook:** [Colab Notebook](https://colab.research.google.com/drive/1qpxrXC3YdJQzly3mRg-4ayYiOjC6rue3?usp=sharing) to experiment and explore.
*   **LLM Examples:**  Dive into specific LLM use cases:
    *   [Llama](examples/models/llama/README.md)
    *   [Qwen 3](examples/models/qwen3/README.md)
    *   [Phi-4-mini](examples/models/phi_4_mini/README.md)
    *   [Llava](examples/models/llava/README.md)

## Community and Contribution

We welcome community engagement and contributions!

*   **Discussion:** Share your feedback, suggestions, and bug reports on the [Discussion Board](https://github.com/pytorch/executorch/discussions).
*   **Discord:** Join the ExecuTorch community on [Discord](https://discord.gg/Dh43CKSAdc) for real-time discussions and support.
*   **Contribution:** Review the [Contributing Guidelines](CONTRIBUTING.md) to learn how to contribute.

## Additional Resources

*   **Codebase Structure:** Refer to the [Codebase structure](CONTRIBUTING.md#codebase-structure) section of the [Contributing Guidelines](CONTRIBUTING.md) for details.
*   **License:** ExecuTorch is BSD licensed; see the [LICENSE](LICENSE) file.