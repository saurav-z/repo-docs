<div align="center">
  <img src="docs/source/_static/img/et-logo.png" alt="ExecuTorch Logo" width="200">
  <h1>ExecuTorch: On-Device AI Framework for Seamless Deployment</h1>
</div>

<div align="center">
  <a href="https://github.com/pytorch/executorch/graphs/contributors"><img src="https://img.shields.io/github/contributors/pytorch/executorch?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/pytorch/executorch/stargazers"><img src="https://img.shields.io/github/stars/pytorch/executorch?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://discord.gg/Dh43CKSAdc"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community"></a>
  <a href="https://pytorch.org/executorch/main/index"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <hr>
</div>

ExecuTorch empowers developers to deploy PyTorch models directly onto a wide array of devices for optimized on-device AI experiences.  Explore the original repository on [GitHub](https://github.com/pytorch/executorch).

## Key Features & Benefits

ExecuTorch provides a powerful and versatile solution for on-device AI, offering:

*   **Broad Platform Support:**  Runs seamlessly on iOS, macOS (ARM64), Android, Linux, and even microcontrollers, enabling wide-ranging deployment options.
*   **Hardware Acceleration:**  Leverages hardware capabilities with support for Apple, Arm, Cadence, MediaTek, NXP, OpenVINO, Qualcomm, Vulkan, and XNNPACK, ensuring optimal performance.
*   **Model Compatibility:** Compatible with a diverse range of models, including LLMs (Large Language Models), CV (Computer Vision), ASR (Automatic Speech Recognition), and TTS (Text to Speech).
*   **Portability:** Enables deployment across diverse computing platforms, from high-end mobile phones to embedded systems and microcontrollers.
*   **Developer Productivity:** Simplifies the development workflow by using the same toolchains and developer tools from PyTorch model authoring and conversion to debugging and deployment across a wide range of platforms.
*   **Performance:** Delivers a high-performance end-user experience with a lightweight runtime, maximizing the use of hardware like CPUs, NPUs, and DSPs.
*   **End-to-End Solution:**  A complete solution for on-device inference and training.

## Getting Started

Ready to deploy your AI models on-device?  Here's how to get started:

*   **Step-by-Step Tutorial:**  Follow the [tutorial](https://pytorch.org/executorch/stable/getting-started.html) to set up ExecuTorch locally and deploy a model to a device.
*   **Interactive Colab Notebook:**  Experiment with ExecuTorch directly in your browser using this [Colab Notebook](https://colab.research.google.com/drive/1qpxrXC3YdJQzly3mRg-4ayYiOjC6rue3?usp=sharing).
*   **LLM Examples:**  Get started with popular open-source LLMs:
    *   [Llama](examples/models/llama/README.md)
    *   [Qwen 3](examples/models/qwen3/README.md)
    *   [Phi-4-mini](examples/models/phi_4_mini/README.md)
    *   [Llava](examples/models/llava/README.md)

## Engage with the Community

Your feedback helps us improve ExecuTorch!

*   **Discussion Board:** Share your thoughts and ideas on the [Discussion Board](https://github.com/pytorch/executorch/discussions).
*   **Discord:**  Chat with the ExecuTorch team and community in real-time on [Discord](https://discord.gg/Dh43CKSAdc).

## Contribute

We welcome contributions from the community. Please review the [contributing guidelines](CONTRIBUTING.md) and join us on [Discord](https://discord.gg/Dh43CKSAdc) to learn more.

## Codebase Structure

See the [Codebase structure](CONTRIBUTING.md#codebase-structure) section within the [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

ExecuTorch is licensed under the BSD license, found in the `LICENSE` file.