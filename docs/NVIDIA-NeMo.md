[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a powerful, cloud-native framework empowering researchers and developers to build, customize, and deploy state-of-the-art generative AI models.  [Explore the NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy cutting-edge LLMs.
*   **Multimodal Models (MMs):** Develop models that process and generate information across multiple modalities.
*   **Automatic Speech Recognition (ASR):** Build accurate and efficient speech recognition systems.
*   **Text-to-Speech (TTS):** Create high-quality speech synthesis models.
*   **Computer Vision (CV):** Develop and deploy advanced computer vision models.
*   **Modular Architecture:** Leverage PyTorch Lightning's modular design for easy experimentation and customization.
*   **Scalable Training:** Train models efficiently across thousands of GPUs.
*   **Optimized for Performance:** Utilize NVIDIA Transformer Engine, Megatron Core, and other techniques for optimized performance.
*   **Deployment and Optimization:** Easily deploy and optimize models using NVIDIA Riva.

## Latest Updates

*   **[Run Hugging Face Models Instantly](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework)**
*   **[Blackwell Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html)**
*   **[Training Performance on GPU Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)**
*   **[New Models Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html)**
*   **[Nemo Framework 2.0](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)**
*   **[New Cosmos World Foundation Models Support](https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform)**

**For more details on these updates, see the original README.**

## Getting Started

1.  **Installation:**

    *   **Conda/Pip:** Install NeMo using `pip install "nemo_toolkit[all]"` in a conda environment.  Refer to the original README for more detailed instructions.
    *   **NGC PyTorch Container:**  For users that want to install from source in a highly optimized container.
    *   **NGC NeMo Container:**  Use pre-built, optimized containers for maximum performance.
2.  **Pre-trained Models:** Access state-of-the-art pre-trained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
3.  **Tutorials & Playbooks:** Follow extensive [tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) and [playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) to get started quickly.
4.  **Example Scripts:** Utilize [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) for advanced training and fine-tuning.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Documentation

*   **Latest:** [https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Stable:** [https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)

## Contribute

We welcome community contributions!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Discussions

Find answers to your questions and engage with the community on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Publications and Blogs

Explore a growing list of [publications](https://nvidia.github.io/NeMo/publications/) and [blogs](https://developer.nvidia.com/blog).

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).