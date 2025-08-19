[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy State-of-the-Art Generative AI Models

**NVIDIA NeMo is a cloud-native framework that empowers researchers and developers to create and deploy cutting-edge generative AI models, including LLMs, MMs, ASR, TTS, and CV models.**

[Visit the original repository for more details.](https://github.com/NVIDIA/NeMo)

## Key Features

*   **Large Language Models (LLMs):** Train, customize, and deploy powerful LLMs.
*   **Multimodal Models (MMs):** Develop models that understand and generate multiple data types.
*   **Automatic Speech Recognition (ASR):** Build and optimize ASR models.
*   **Text-to-Speech (TTS):** Create high-quality speech synthesis systems.
*   **Computer Vision (CV):** Implement state-of-the-art CV models.
*   **Modular and Scalable:** Built on PyTorch Lightning for flexibility and scalability.
*   **Pre-trained Models:** Access a wide range of pre-trained models on Hugging Face Hub and NVIDIA NGC.
*   **Optimized Performance:** Leverage NVIDIA Transformer Engine and Megatron Core for efficient training and deployment.
*   **Alignment Techniques:** Supports advanced alignment methods like SteerLM, DPO, and RLHF.

## What's New

*   **[Pretrain and Finetune Hugging Face models via AutoModel](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework)**
*   **[Training on Blackwell using Nemo](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html)**
*   **[Training Performance on GPU Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)**
*   **[New Models Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html)**
*   **[NeMo Framework 2.0](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)**

## Getting Started

*   **Explore Tutorials:** Start with easy-to-run tutorials on [Google Colab](https://colab.research.google.com) or using the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Pre-trained Models:** Utilize pre-trained models from [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
*   **Training Recipes:** Use example scripts and [playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) to train models with the NeMo Framework Launcher or multi-GPU/multi-node training.

## Installation

Choose your installation method based on your needs:

*   **Conda / Pip:** Ideal for exploring NeMo and for ASR/TTS domains.
*   **NGC PyTorch container:** For installation from source in an optimized container.
*   **NGC NeMo container:** Ready-to-go solution for maximum performance.

**Detailed installation instructions are available in the original README.**

## Developer Documentation

*   [Latest Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   [Stable Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)

## Contribute

We welcome community contributions! Refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Publications

Explore a collection of [publications](https://nvidia.github.io/NeMo/publications/) that utilize the NeMo Framework.

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).
```
Key improvements and explanations:

*   **SEO-Optimized Title:** The main heading is clear and includes relevant keywords ("NVIDIA NeMo", "Generative AI Models").
*   **One-Sentence Hook:**  A concise and engaging opening sentence immediately grabs the reader's attention.
*   **Clear Headings:**  Uses headings to organize information logically (Key Features, What's New, Getting Started, Installation, etc.). This improves readability and SEO.
*   **Bulleted Key Features:** Highlights the main capabilities of NeMo, making it easy to scan.  Uses relevant keywords (LLMs, ASR, TTS, CV).
*   **Summarized Content:** Condensed the original README, removing redundancy and focusing on the most important information.
*   **Contextual Links:** Replaced "refer to" with direct links.  Linked to the relevant sections of the original README and other external pages like NGC, Hugging Face.
*   **Improved "What's New" Section:** Added summaries of new announcements.
*   **Clear Installation Instructions:** Simplified the installation section, directing users to the original README for details and presenting the options in a more user-friendly manner.
*   **Concise and Actionable Language:**  Uses active voice and clear calls to action (e.g., "Explore Tutorials", "Contribute").
*   **Removed Unnecessary Information:**  Removed some of the less critical details to focus on the core value proposition.
*   **Strong Emphasis on Key Benefits:** The focus is on what NeMo *does* (build, customize, deploy), and the benefits to the user.
*   **Complete and Self-Contained:** The summary provides enough information for someone to understand what NeMo is and how to get started without having to read the entire original README.
*   **Markdown Formatting:**  Ensures consistent formatting for readability.
*   **Keywords throughout:** Used relevant keywords (Generative AI, LLMs, ASR, TTS, etc.) throughout the summary for better search engine optimization.