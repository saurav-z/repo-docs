[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build and Deploy Generative AI Models with Ease

NVIDIA NeMo is a powerful, cloud-native framework designed to accelerate the development of Large Language Models (LLMs), multimodal models, and more, and its source code is hosted on [GitHub](https://github.com/NVIDIA/NeMo).

## Key Features

*   **LLMs:**  Develop and customize state-of-the-art Large Language Models.
*   **Multimodal Models:** Build models that process and generate content across multiple modalities (e.g., text, images, video).
*   **ASR & TTS:**  Leverage advanced Automatic Speech Recognition and Text-to-Speech capabilities.
*   **Computer Vision:** Explore and implement cutting-edge computer vision techniques.
*   **Scalability:** Train models efficiently across thousands of GPUs with NeMo-Run.
*   **Modular Design:** Benefit from a flexible, modular architecture.
*   **Pre-trained Models:** Utilize pre-trained models available on Hugging Face Hub and NVIDIA NGC to accelerate your projects.
*   **Deployment:** Optimize and deploy models with NVIDIA Riva and NeMo Microservices.
*   **Training Techniques:** Utilize advanced techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, MoE, and Mixed Precision Training.
*   **Parameter-Efficient Fine-tuning (PEFT):** Leverage LoRA, P-Tuning, Adapters, and IA3 for efficient model customization.

## Latest Updates

*   **[New Models Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html):** Support for models like Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **[Training on Blackwell](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html):** Performance benchmarks on GB200 & B200.
*   **[Pretrain and finetune Hugging Face models via AutoModel](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework):** AutoModel support for Hugging Face models.

*   **[NVIDIA Cosmos World Foundation Models](https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform):** Integration with Cosmos platform for physical AI.

## Getting Started

NeMo provides various ways to get started:

*   **Tutorials:** Run tutorials on Google Colab or with the NGC NeMo Framework Container.
*   **Pre-trained Models:** Access readily available pre-trained models on Hugging Face Hub and NVIDIA NGC.
*   **Example Scripts:**  Utilize example scripts for multi-GPU/multi-node training.
*   **Playbooks:** Use the NeMo Framework Launcher for training NeMo models with ease.

## Installation

Choose the installation method that suits your needs:

*   **Conda / Pip:** Install with pip into a virtual environment (recommended for ASR and TTS).
*   **NGC PyTorch Container:** Install from source within a highly optimized NVIDIA PyTorch container.
*   **NGC NeMo Container:** Use a ready-to-go container for optimal performance.

**Requirements:**
*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for model training)

### Detailed Installation Instructions

Follow these steps to install the NeMo framework using Conda / Pip.
```bash
conda create --name nemo python==3.10.12
conda activate nemo
pip install "nemo_toolkit[all]"
```
If a specific version is desired, we recommend a Pip-VCS install. From [NVIDIA/NeMo](github.com/NVIDIA/NeMo), fetch the commit, branch, or tag that you would like to install.  
To install nemo_toolkit from this Git reference `$REF`, use the following installation method:
```bash
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout @${REF:-'main'}
pip install '.[all]'
```
To install a specific domain of NeMo, you must first install the
nemo_toolkit using the instructions listed above. Then, you run the
following domain-specific commands:
```bash
pip install nemo_toolkit['all'] # or pip install "nemo_toolkit['all']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['asr'] # or pip install "nemo_toolkit['asr']@git+https://github.com/NVIDIA/NeMo@$REF:-'main'}"
pip install nemo_toolkit['nlp'] # or pip install "nemo_toolkit['nlp']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['tts'] # or pip install "nemo_toolkit['tts']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['vision'] # or pip install "nemo_toolkit['vision']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['multimodal'] # or pip install "nemo_toolkit['multimodal']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
```

## Resources

*   **Documentation:**  [NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Discussions:**  [NeMo Discussions Board](https://github.com/NVIDIA/NeMo/discussions)
*   **Publications:**  [Publications](https://nvidia.github.io/NeMo/publications/)
*   **Contribute:**  [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md)

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).