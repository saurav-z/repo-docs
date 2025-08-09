<div align="center">
  <img src="docs/source/_static/img/et-logo.png" alt="ExecuTorch Logo" width="200">
  <h1>ExecuTorch: Powering On-Device AI with High Performance</h1>
</div>

<div align="center">
  <a href="https://github.com/pytorch/executorch/graphs/contributors"><img src="https://img.shields.io/github/contributors/pytorch/executorch?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/pytorch/executorch/stargazers"><img src="https://img.shields.io/github/stars/pytorch/executorch?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://discord.gg/Dh43CKSAdc"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://pytorch.org/executorch/main/index"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <hr>
</div>

ExecuTorch is a cutting-edge, end-to-end solution from Meta designed to revolutionize on-device AI experiences, enabling high-performance inference and training directly on a wide range of devices. ([See the original repo](https://github.com/pytorch/executorch))

**Key Features of ExecuTorch:**

*   **Broad Model Support:** Compatible with a diverse range of AI models, including:
    *   Large Language Models (LLMs)
    *   Computer Vision (CV) models
    *   Automatic Speech Recognition (ASR) models
    *   Text-to-Speech (TTS) models
*   **Extensive Platform Support:** Runs on various operating systems and hardware:
    *   **Operating Systems:** iOS, macOS (ARM64), Android, Linux, Microcontrollers
    *   **Hardware Acceleration:** Apple, Arm, Cadence, MediaTek, NXP, OpenVINO, Qualcomm, Vulkan, XNNPACK
*   **Portability:**  Runs across a wide spectrum of devices, from high-end mobile phones to resource-constrained embedded systems and microcontrollers.
*   **Developer Productivity:**  Empowers developers with familiar PyTorch tools and streamlined workflows for model authoring, conversion, debugging, and deployment.
*   **High Performance:** Delivers a seamless user experience with a lightweight runtime that leverages the full capabilities of hardware accelerators like CPUs, NPUs, and DSPs.

## Getting Started

Quickly get started with ExecuTorch:

*   **Step-by-Step Tutorial:**  Learn how to get things running locally and deploy a model to a device by visiting the [Step by Step Tutorial](https://pytorch.org/executorch/stable/getting-started.html).
*   **Interactive Colab Notebook:** Experiment and explore ExecuTorch directly in your browser using this [Colab Notebook](https://colab.research.google.com/drive/1qpxrXC3YdJQzly3mRg-4ayYiOjC6rue3?usp=sharing).
*   **LLM Examples:** Dive into LLM use cases with specific instructions for popular open-source models such as [Llama](examples/models/llama/README.md), [Qwen 3](examples/models/qwen3/README.md), [Phi-4-mini](examples/models/phi_4_mini/README.md), and [Llava](examples/models/llava/README.md).

## Community and Contribution

We encourage community feedback and contributions.

*   **Discussion:** Share your ideas and questions on the [Discussion Board](https://github.com/pytorch/executorch/discussions).
*   **Real-time Chat:** Connect with the community on [Discord](https://discord.gg/Dh43CKSAdc).
*   **Contributing Guidelines:** Review the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute.

## Directory Structure

Refer to the [Codebase structure](CONTRIBUTING.md#codebase-structure) section of the [Contributing Guidelines](CONTRIBUTING.md) for detailed information.

## License

ExecuTorch is licensed under the BSD license; see the `LICENSE` file for details.