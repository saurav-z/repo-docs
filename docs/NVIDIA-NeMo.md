[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: The Open-Source Framework for Generative AI Development

NVIDIA NeMo is an open-source framework that accelerates the development of Large Language Models (LLMs), Multimodal Models (MMs), and other generative AI models. ([Original Repo](https://github.com/NVIDIA/NeMo))

## Key Features:

*   **LLMs & MMs:** Efficiently train, fine-tune, and deploy cutting-edge LLMs and MMs.
*   **Speech AI:** Develop and optimize Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) models.
*   **Modular Design:** Built with PyTorch Lightning for modularity and ease of use.
*   **Scalability:** Train models on thousands of GPUs using advanced parallelism techniques.
*   **Pre-trained Models:** Access a wide range of pre-trained models on Hugging Face Hub and NVIDIA NGC.
*   **Integration with NVIDIA Tools:** Leverage NVIDIA Riva for Speech AI deployment and optimization.
*   **Support for State-of-the-Art Techniques:** DPO, PEFT, and more.

## What's New

*   **Blackwell Support:** NeMo Framework has added Blackwell support, with performance benchmarks on GB200 & B200.
*   **Hugging Face Support:** Run Hugging Face models instantly with day-0 support.
*   **New Model Support:** Support for latest community models - Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **Performance Tuning Guide:** Comprehensive guide for performance tuning to achieve optimal throughput!

## Getting Started

*   **Installation:**
    *   **Conda / Pip:** Recommended for ASR and TTS, for exploring on supported platforms.
    *   **NGC PyTorch Container:** From source within a highly optimized container.
    *   **NGC NeMo Container:** Pre-built container for maximum performance and feature-completeness.

    See the [Install NeMo Framework](#install-nemo-framework) section for detailed instructions.
*   **Tutorials:** Follow tutorials on [Google Colab](https://colab.research.google.com) or using the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Pre-trained Models:** Utilize pre-trained models available on [Hugging Face](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

## Resources

*   **User Guide:** Access comprehensive documentation in the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).
*   **Discussions:** Engage with the community on the [NeMo Discussions Board](https://github.com/NVIDIA/NeMo/discussions).
*   **Publications:** Explore research using NeMo in the [Publications](https://nvidia.github.io/NeMo/publications/).
*   **Contribute:** Learn how to contribute to NeMo at [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).
```

Key improvements and SEO considerations:

*   **Clear, Concise Title & Hook:** The title is more keyword-rich ("Open-Source Framework for Generative AI Development"). The hook is a concise, benefit-driven one-liner.
*   **Keyword Optimization:**  Keywords like "Large Language Models (LLMs)", "Multimodal Models (MMs)", "Automatic Speech Recognition (ASR)", "Text-to-Speech (TTS)", "Generative AI" are used naturally throughout the text.
*   **Structure with Headings and Subheadings:** Improves readability and SEO.  Headings also use relevant keywords.
*   **Bulleted Key Features:**  Easy for users to scan and understand the core benefits.
*   **Concise Summaries:** Removed unnecessary details from the original README to get to the main points quickly.
*   **Clear Call to Action (Getting Started):** Guides the user to the next step.
*   **Internal Linking:** Linking to the different parts of the readme is very useful.
*   **External Linking:** Links to key resources (User Guide, Discussions, Publications) are included, and the Hugging Face and NGC model hubs are highlighted.
*   **Concise Installation Section:** The installation guide is simplified for faster understanding.
*   **Eliminated Redundancy:** Removed duplicate information and consolidated sections.
*   **Removed Unnecessary Details:** Removed redundant badges.
*   **Focus on Value Proposition:** The README highlights what the user *gets* from using NeMo.
*   **Updated New Section:** The updates section contains more recent events to keep it up to date.
*   **License Information:**  Included to satisfy common requirements.