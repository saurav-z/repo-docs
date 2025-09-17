# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models

NVIDIA NeMo is a cloud-native framework empowering researchers and developers to create cutting-edge generative AI models efficiently.  [Explore the NeMo Repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy state-of-the-art LLMs.
*   **Multimodal Models (MMs):** Develop models that process and generate multiple data types (text, images, etc.).
*   **Automatic Speech Recognition (ASR):** Build high-accuracy ASR models.
*   **Text-to-Speech (TTS):** Create realistic and expressive speech synthesis.
*   **Computer Vision (CV):** Implement advanced computer vision tasks.
*   **Modular Architecture:** Leverages PyTorch Lightning for flexibility and ease of use.
*   **Scalability:** Supports distributed training across thousands of GPUs with NeMo-Run and various parallelism strategies.
*   **Pre-trained Models:** Access a wide range of pre-trained models on Hugging Face Hub and NVIDIA NGC.
*   **Model Optimization & Deployment**: Optimize LLMs & MMs with NVIDIA NeMo Microservices and ASR & TTS models with NVIDIA Riva.

## Getting Started

### What's New: Highlights of Recent Updates

*   **Support for Hugging Face Models:** NeMo's AutoModel feature now supports various Hugging Face models, including AutoModelForCausalLM and AutoModelForImageTextToText, offering day-0 support.
*   **Blackwell Support:** Performance benchmarks on GB200 & B200 are available to leverage the efficiency of Blackwell.
*   **New Model Support:** Includes support for Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo 2.0:** Prioritizing modularity and ease-of-use with Python-based configuration, modular abstractions, and scalability.
*   **Cosmos World Foundation Models:** Support for training and customizing the NVIDIA Cosmos collection, enhancing video generation capabilities.
*   **Speech Recognition Enhancements:** Up to 10x inference speed-up for speech recognition models.

### Installation

Choose your preferred method:

*   **Conda / Pip:** Recommended for ASR and TTS domains and exploring NeMo.
    ```bash
    conda create --name nemo python==3.10.12
    conda activate nemo
    pip install "nemo_toolkit[all]"
    ```
*   **NGC PyTorch Container:** Install from source in a highly optimized container.
*   **NGC NeMo Container:** Ready-to-use solution for optimal performance.

**Note:** Full support is provided for Linux (amd64/x86_64) installations via NGC PyTorch container. Refer to the table for full support information.

## Resources

*   [**Documentation:**](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/) Comprehensive user guide.
*   [**Quickstart:**](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) Get started with NeMo 2.0.
*   [**Tutorials:**](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) Run tutorials on Google Colab or with the NGC NeMo Framework Container.
*   [**Example Scripts:**](https://github.com/NVIDIA/NeMo/tree/main/examples) Multi-GPU/multi-node training examples.
*   [**Publications:**](https://nvidia.github.io/NeMo/publications/) List of publications using NeMo.

## Contribute

We welcome contributions!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).