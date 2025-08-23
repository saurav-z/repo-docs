[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a cloud-native framework empowering researchers and developers to create state-of-the-art generative AI models for LLMs, MMs, ASR, TTS, and CV.  [Explore the NeMo Framework on GitHub](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):**  Train and fine-tune LLMs with cutting-edge techniques.
*   **Multimodal Models (MMs):** Develop models that combine different data types.
*   **Automatic Speech Recognition (ASR):** Build and optimize ASR models for various applications.
*   **Text-to-Speech (TTS):**  Create high-quality speech synthesis systems.
*   **Computer Vision (CV):** Develop and deploy advanced computer vision models.
*   **Modular and Scalable:** Built on PyTorch Lightning for flexibility and scalability across thousands of GPUs.
*   **Optimized Training:** Leverages techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, MoE, and mixed precision training.
*   **Deployment & Optimization:**  Integrates with NVIDIA Riva for optimized inference and deployment.
*   **Parameter Efficient Fine-tuning (PEFT):** Supports LoRA, P-Tuning, Adapters, and IA3.
*   **Cosmos Integration:**  Supports training and customizing NVIDIA Cosmos world foundation models.

## What's New

*   **25.04 Updates:** Broad support for Hugging Face models via AutoModel, particularly AutoModelForCausalLM and AutoModelForImageTextToText.
*   **Blackwell Support:** Performance benchmarks on GB200 & B200.
*   **Performance Guide:** Comprehensive guide for performance tuning.
*   **New Model Support:** Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.

For the latest news, see the [NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## Get Started

*   **Quickstart:** [Nemo-Run Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html).
*   **User Guide:** [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
*   **Recipes:** [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes).
*   **Feature Guide:** [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
*   **Migration Guide:** [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide).
*   **Cosmos:** [NeMo Curator](https://developer.nvidia.com/nemo-curator).
*   **Cosmos:** [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## Installation

Choose your installation method based on your needs:

*   **Conda / Pip:**  Suitable for exploring NeMo and recommended for ASR and TTS.

    ```bash
    conda create --name nemo python==3.10.12
    conda activate nemo
    pip install "nemo_toolkit[all]"
    ```
*   **NGC PyTorch Container:** Install from source within a highly optimized container (requires NVIDIA PyTorch container).
    ```bash
    docker run \
      --gpus all \
      -it \
      --rm \
      --shm-size=16g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/pytorch:25.01-py3'}
    ```

    ```bash
    cd /opt
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout ${REF:-'main'}
    bash docker/common/install_dep.sh --library all
    pip install ".[all]"
    ```
*   **NGC NeMo Container:**  Ready-to-use, pre-built containers for optimal performance.

    ```bash
    docker run \
      --gpus all \
      -it \
      --rm \
      --shm-size=16g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'}
    ```

For domain-specific installations, refer to the detailed instructions in the original README.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## Resources

*   **Developer Documentation:** [Latest Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).
*   **Discussions Board:**  [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions) - ask questions and engage with the community.
*   **Publications:** [Publications](https://nvidia.github.io/NeMo/publications/) - Explore research papers using NeMo.
*   **Blogs:** See the original README for links to relevant NVIDIA blogs.
*   **Contributing:** [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) - Learn how to contribute to NeMo.

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).