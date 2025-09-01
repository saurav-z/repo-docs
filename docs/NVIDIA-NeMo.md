[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

**NVIDIA NeMo is a cloud-native, open-source framework that simplifies the creation, customization, and deployment of cutting-edge generative AI models.**

[View the original repository on GitHub](https://github.com/NVIDIA/NeMo)

## Key Features:

*   **Comprehensive Domain Support:** Built for Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Modular and Flexible:** Designed for researchers and PyTorch developers to efficiently build and experiment with new AI models.
*   **Scalable Training:**  Leverages PyTorch Lightning and NeMo-Run for training across thousands of GPUs.
*   **Pre-trained Models & Customization:**  Access state-of-the-art pre-trained models and easily customize them using existing code.
*   **Optimized Deployment:** Deploy and optimize your models with NVIDIA Riva and NeMo Microservices.
*   **Cutting-Edge Training Techniques:**  Supports advanced techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, MoE, BFloat16/FP8 mixed precision.
*   **Integration with NVIDIA Technologies:**  Utilizes NVIDIA Transformer Engine and Megatron Core for performance.
*   **Alignment and Fine-tuning:** Supports SteerLM, DPO, RLHF, LoRA, P-Tuning, Adapters, and IA3.
*   **Cosmos Integration:** Includes support for video dataset curation and post-training of the Cosmos World Foundation Models.

## Latest News:

*   **Pretrain and finetune :hugs:Hugging Face models via AutoModel**
    *   NeMo Framework's latest feature AutoModel enables broad support for :hugs:Hugging Face models, with 25.04 focusing on
        *   `AutoModelForCausalLM` in the Text Generation category
        *   `AutoModelForImageTextToText` in the Image-Text-to-Text category
    *   More Details in Blog: [Run Hugging Face Models Instantly with Day-0 Support from NVIDIA NeMo Framework](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework)
*   **Training on Blackwell using Nemo**
    *   NeMo Framework has added Blackwell support, with performance benchmarks on GB200 & B200. More optimizations to come in the upcoming releases.
*   **Training Performance on GPU Tuning Guide**
    *   NeMo Framework has published [a comprehensive guide for performance tuning to achieve optimal throughput](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)!
*   **New Models Support**
    *   NeMo Framework has added support for latest community models - Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.

*For older updates, see the original README.*

## Getting Started

*   **Documentation:**  Comprehensive user guides are available at [NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).
*   **Quickstart:** Examples of using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster.
*   **NeMo 2.0 Documentation:** Find more information about NeMo 2.0 in the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
*   **Recipes:**  [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) contains additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run.
*   **Feature Guide:** For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
*   **Migration Guide:**  To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.
*   **Cosmos:** For more information on video datasets, refer to [NeMo Curator](https://developer.nvidia.com/nemo-curator). To post-train World Foundation Models using the NeMo Framework for your custom physical AI tasks, see the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## Installation

Choose your installation method:

*   **Conda / Pip:**  Install with `pip install "nemo_toolkit[all]"` or pip install specific domains, recommended for ASR and TTS, explore NeMo on any supported platform.
*   **NGC PyTorch container:** Install from source into a highly optimized container.
*   **NGC NeMo container:** Ready-to-go solution, designed for maximum performance.

*Refer to the original README for detailed instructions on each installation method.*

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## Documentation and Resources

*   **Developer Documentation:** See [NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Publications:** View a curated list of publications at [publications](https://nvidia.github.io/NeMo/publications/)
*   **Discussions Board:**  Ask questions and join discussions on the [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Contributing

Contributions are welcome!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## License

Licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).