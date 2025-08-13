[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a powerful, cloud-native framework empowering researchers and developers to build and deploy state-of-the-art generative AI models across various domains, from large language models (LLMs) to multimodal applications.  For more detailed information, visit the [original NeMo repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):**  Build and customize powerful LLMs.
*   **Multimodal Models (MMs):**  Develop AI that understands and generates multiple data types.
*   **Automatic Speech Recognition (ASR):**  Accurately transcribe speech.
*   **Text to Speech (TTS):**  Generate high-quality speech from text.
*   **Computer Vision (CV):**  Leverage advanced computer vision capabilities.
*   **Modular Design:** Simplify model adaptation and experimentation with PyTorch Lightning's modular abstractions.
*   **Scalability:**  Train models efficiently on thousands of GPUs using NeMo-Run and other advanced techniques.
*   **Pre-trained Models:** Access a wide range of pre-trained models on Hugging Face Hub and NVIDIA NGC.
*   **Deployment & Optimization:** Deploy and optimize your models with NVIDIA NeMo Microservices and Riva.
*   **Parameter Efficient Fine-tuning:** Utilize techniques like LoRA, P-Tuning, and Adapters for efficient model customization.
*   **Full support for all new models:** Hyena, Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.

## What's New

*   **NeMo 2.0:** Prioritizes modularity and ease-of-use, offering Python-based configuration, and improved modular abstractions.
*   **Blackwell Support:** Added Blackwell support with performance benchmarks on GB200 & B200.
*   **Hugging Face Model Support:** Instantly run Hugging Face models via AutoModel support.
*   **Cosmos World Foundation Models Support:** Now supports training and customizing NVIDIA Cosmos models.

## Getting Started

NeMo provides several ways to get started, including:

*   **Pre-trained Models:**  Explore pre-trained models available on [Hugging Face](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
*   **Tutorials:**  Run tutorials on [Google Colab](https://colab.research.google.com) or using the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Playbooks:** Train NeMo models with the [NeMo Framework Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).
*   **Example Scripts:** Customize models from scratch or fine-tune existing ones using a suite of [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples).

## Installation

Choose the installation method that best fits your needs:

*   **Conda / Pip:** Install NeMo using `pip install "nemo_toolkit[all]"` within a Conda environment.  This is the recommended method for ASR and TTS domains.
*   **NGC PyTorch container:** Install from source within an optimized NVIDIA PyTorch container.
*   **NGC NeMo container:** Use a pre-built, optimized container for highest performance.

See the [Installation](#install-nemo-framework) section for detailed instructions and platform support.

## Developer Documentation

Comprehensive documentation is available:

*   **Latest:** [Documentation of the latest branch](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Stable:** [Documentation of the most recent release](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)

## Contribute

We welcome community contributions.  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Publications & Blogs

Explore a growing list of [publications](https://nvidia.github.io/NeMo/publications/) and [blogs](#blogs) utilizing the NeMo Framework.

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).
```
Key improvements and SEO optimizations:

*   **Clear, Concise Headline:**  Uses a strong, SEO-friendly title with a relevant keyword ("NVIDIA NeMo") and action verbs ("Build, Customize, and Deploy").
*   **One-Sentence Hook:** The introductory sentence concisely explains the framework's purpose.
*   **Bulleted Key Features:** Uses bullet points for easy readability and keyword inclusion.
*   **Subheadings:**  Organizes information logically with clear subheadings for better scannability and SEO.
*   **Keywords:**  Strategically incorporates relevant keywords throughout the text (e.g., "Large Language Models (LLMs)," "Multimodal Models," "Generative AI," "Automatic Speech Recognition").
*   **Links:**  Includes internal links to relevant sections (e.g., "Installation," "Getting Started") and external links to the original repo and documentation.
*   **Concise Summaries:**  Condenses information from the original README while maintaining clarity.
*   **Call to Action:**  Encourages exploration of pre-trained models, tutorials, and example scripts.
*   **Focus on Benefits:** Highlights the benefits of using NeMo, such as ease of use, scalability, and pre-trained models.
*   **Modern Formatting:** Uses a clean and modern markdown style.
*   **Removed redundant information**  Removed redundant information, like the "latest news" and "blogs" sections, which were repetitive.  Incorporated the most important information within the rest of the summary.
*   **Reordered content:** Optimized the order of the information, placing the most important information at the top.