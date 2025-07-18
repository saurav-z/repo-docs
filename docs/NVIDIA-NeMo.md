[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: The Comprehensive Framework for Generative AI 

**NVIDIA NeMo is a cloud-native framework for building, training, and deploying cutting-edge generative AI models, including LLMs, MMs, ASR, and TTS.**  ([View the original repo](https://github.com/NVIDIA/NeMo))

## Key Features

*   **Large Language Models (LLMs):** Develop and customize powerful language models.
*   **Multimodal Models (MMs):** Create AI models that process and generate multiple data types (text, image, audio, etc.).
*   **Automatic Speech Recognition (ASR):** Build and optimize speech-to-text systems.
*   **Text-to-Speech (TTS):** Generate natural-sounding speech from text.
*   **Computer Vision (CV):** Develop and deploy computer vision models.
*   **Scalable Training:** Easily scale model training across thousands of GPUs.
*   **Model Optimization & Deployment:** Optimize models for inference and deploy with NVIDIA Riva and NeMo Microservices.
*   **Pre-trained Models:** Leverage state-of-the-art, pre-trained models available on Hugging Face Hub and NVIDIA NGC.
*   **Parameter Efficient Fine-tuning (PEFT):** Support for techniques like LoRA, P-Tuning, and Adapters.
*   **Hybrid State Space Model Support:** Utilizing NVIDIA Megatron Core for scaling Transformer model training

## What's New

*   **Hugging Face Integration:**  Seamlessly integrate and fine-tune Hugging Face models using AutoModel.
*   **Blackwell Support:** Enhanced performance benchmarks on GB200 & B200
*   **Training Performance Guide:** A comprehensive guide for optimal throughput is now published.
*   **New Model Support:** Including Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, and Qwen3-30B&32B.
*   **NeMo 2.0:**  Focuses on modularity and ease-of-use.

## Getting Started

### Installation

Choose your preferred method:

*   **Conda / Pip:**
    *   Create a Conda environment: `conda create --name nemo python==3.10.12`
    *   Activate the environment: `conda activate nemo`
    *   Install with pip: `pip install "nemo_toolkit[all]"`
    *   Install a specific domain: `pip install nemo_toolkit['asr']` (replace 'asr' with 'nlp', 'tts', 'vision', or 'multimodal' as needed)

*   **NGC PyTorch Container:**
    1.  Start with an NVIDIA PyTorch container (e.g., `nvcr.io/nvidia/pytorch:25.01-py3`).
    2.  Run the container and install Nemo from source:
        ```bash
        docker run --gpus all -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/pytorch:25.01-py3'}
        cd /opt
        git clone https://github.com/NVIDIA/NeMo
        cd NeMo
        git checkout ${REF:-'main'}
        bash docker/common/install_dep.sh --library all
        pip install ".[all]"
        ```

*   **NGC NeMo Container:**
    *   Run the pre-built container:
        ```bash
        docker run --gpus all -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'}
        ```

### Resources

*   **Documentation:** [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Tutorials:** Run tutorials on [Google Colab](https://colab.research.google.com) or with the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Example Scripts:** Explore example scripts for advanced users.
*   **Discussions:** Find answers and engage with the community on the [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Contributing

We welcome community contributions!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).
```
Key improvements and SEO optimizations:

*   **Concise Hook:**  The one-sentence hook immediately grabs attention and clearly states the framework's purpose.
*   **Keyword-Rich Headings:** Includes keywords like "Generative AI," "LLMs," "MMs," "ASR," and "TTS" in headings and subheadings to improve search ranking.
*   **Bulleted Key Features:** Uses bullet points for easy readability and highlights the most important aspects of NeMo.
*   **Clear Structure:**  Organizes the information logically, making it easy to understand.
*   **Actionable Instructions:** Provides clear installation instructions with multiple options.
*   **Emphasis on Benefits:** Highlights the benefits of using NeMo (e.g., scalability, pre-trained models).
*   **Internal Linking:** Links to the original repository and the relevant documentation for improved navigation.
*   **Concise Language:** Removes unnecessary jargon and focuses on clarity.
*   **Clean Formatting:** Uses consistent formatting (bold text, lists) for better visual appeal.
*   **Optimized Sections:** Condenses and summarizes information from the original README, removing excessive detail.
*   **Updated Information:** Refers to the most recent releases and features.