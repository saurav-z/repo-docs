<div align="center">
  <img src="docs/source/_static/img/et-logo.png" alt="ExecuTorch Logo" width="200">
  <h1>ExecuTorch: Unleash the Power of On-Device AI</h1>
</div>

<div align="center">
  <a href="https://github.com/pytorch/executorch/graphs/contributors"><img src="https://img.shields.io/github/contributors/pytorch/executorch?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/pytorch/executorch/stargazers"><img src="https://img.shields.io/github/stars/pytorch/executorch?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://discord.gg/Dh43CKSAdc"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://pytorch.org/executorch/main/index"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <hr>
</div>

ExecuTorch is a cutting-edge, open-source framework from PyTorch, enabling high-performance on-device AI experiences across various platforms. ([See the original repository](https://github.com/pytorch/executorch))

## Key Features & Benefits

ExecuTorch empowers developers to deploy and run AI models directly on devices, offering significant advantages:

*   **Wide Platform Support:**
    *   iOS
    *   MacOS (ARM64)
    *   Android
    *   Linux
    *   Microcontrollers
*   **Hardware Acceleration:** Optimized for various hardware including:
    *   Apple
    *   Arm
    *   Cadence
    *   MediaTek
    *   NXP
    *   OpenVINO
    *   Qualcomm
    *   Vulkan
    *   XNNPACK
*   **Portability:**  Deploy your models across a diverse range of hardware, from mobile phones to embedded systems.
*   **Productivity:**  Leverage familiar PyTorch tools and workflows for model authoring, conversion, debugging, and deployment.
*   **Performance:**  Experience high-performance inference and training with a lightweight runtime that utilizes CPU, NPU, and DSP capabilities.
*   **Model Compatibility:** Supports a broad range of model types, including:
    *   Large Language Models (LLMs)
    *   Computer Vision (CV)
    *   Automatic Speech Recognition (ASR)
    *   Text-to-Speech (TTS)

## Getting Started

Dive into on-device AI development with these resources:

*   [Step by Step Tutorial](https://pytorch.org/executorch/stable/getting-started.html): Get up and running locally and deploy a model to a device.
*   [Colab Notebook](https://colab.research.google.com/drive/1qpxrXC3YdJQzly3mRg-4ayYiOjC6rue3?usp=sharing): Experiment with ExecuTorch directly in your browser.
*   **LLM Examples:** Explore pre-built examples for popular open-source models:
    *   [Llama](examples/models/llama/README.md)
    *   [Qwen 3](examples/models/qwen3/README.md)
    *   [Phi-4-mini](examples/models/phi_4_mini/README.md)
    *   [Llava](examples/models/llava/README.md)

## Contribute and Connect

We encourage community involvement!

*   [Discussion Board](https://github.com/pytorch/executorch/discussions): Share feedback, suggestions, and report issues.
*   [Discord](https://discord.gg/Dh43CKSAdc): Chat with the community and the development team in real-time.
*   [Contributing Guidelines](CONTRIBUTING.md): Learn how to contribute code, documentation, and more.

## Directory Structure

For details on the codebase structure, refer to the [Codebase structure](CONTRIBUTING.md#codebase-structure) section of the [Contributing Guidelines](CONTRIBUTING.md).

## License

ExecuTorch is licensed under the BSD license. See the `LICENSE` file for details.