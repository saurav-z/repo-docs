[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a cloud-native framework empowering researchers and developers to build, customize, and deploy state-of-the-art generative AI models for LLMs, MMs, ASR, TTS, and Computer Vision.  [Explore the NeMo repository](https://github.com/NVIDIA/NeMo).

## Key Features:

*   **Comprehensive Support:** Built for Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Modular Architecture:** NeMo 2.0 introduces Python-based configuration and modular abstractions for flexible model development.
*   **Scalable Training:** Seamlessly scale experiments across thousands of GPUs using NeMo-Run and advanced distributed training techniques.
*   **Pre-trained Models:** Leverage pre-trained models available on Hugging Face Hub and NVIDIA NGC for rapid prototyping.
*   **Model Customization:** Fine-tune and align LLMs with cutting-edge techniques like SteerLM, DPO, and RLHF, and PEFT methods such as LoRA.
*   **Deployment & Optimization:** Deploy and optimize NeMo models with NVIDIA NeMo Microservices and Riva.
*   **Multi-Platform Support:** Flexible installation options via Conda/Pip, NGC PyTorch containers, and NGC NeMo containers.

## Latest News & Updates

*   **[2025-05-19] Pretrain and finetune Hugging Face models via AutoModel:** NeMo Framework's latest feature AutoModel enables broad support for Hugging Face models, with 25.04 focusing on AutoModelForCausalLM and AutoModelForImageTextToText.
*   **[2025-05-19] Training on Blackwell using Nemo:** NeMo Framework has added Blackwell support, with performance benchmarks on GB200 & B200.
*   **[2025-05-19] Training Performance on GPU Tuning Guide:** NeMo Framework has published a comprehensive guide for performance tuning to achieve optimal throughput!
*   **[2025-05-19] New Models Support:** NeMo Framework has added support for latest community models including Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **[2025-05-19] NeMo Framework 2.0:**  Release of NeMo 2.0, which prioritizes modularity and ease-of-use.
*   **[2025-01-09] New Cosmos World Foundation Models Support** Advancing Physical AI with NVIDIA Cosmos World Foundation Model Platform.
*   **[2025-01-07] Accelerate Custom Video Foundation Model Pipelines with New NVIDIA NeMo Framework Capabilities** The NeMo Framework now supports training and customizing the NVIDIA Cosmos collection of world foundation models and NeMo Curator.
*   **[2024-11-06] Large Language Models and Multimodal Models**  NVIDIA recently announced significant enhancements to the NeMo platform, focusing on multimodal generative AI models with NeMo Curator and the Cosmos tokenizer.
*   **[2024-07-23] Llama 3.1 Support:** The NeMo Framework now supports training and customizing the Llama 3.1 collection of LLMs from Meta.
*   **[2024-07-16] Accelerate your Generative AI Distributed Training Workloads with the NVIDIA NeMo Framework on Amazon EKS:** NVIDIA NeMo Framework now runs distributed training workloads on an Amazon Elastic Kubernetes Service (Amazon EKS) cluster.
*   **[2024/06/17] NVIDIA NeMo Accelerates LLM Innovation with Hybrid State Space Model Support** NVIDIA NeMo and Megatron Core now support pre-training and fine-tuning of state space models (SSMs).
*   **[2024-06-18] NVIDIA releases 340B base, instruct, and reward models** NVIDIA releases 340B base, instruct, and reward models pretrained on a total of 9T tokens.
*   **[2024/06/12] NVIDIA sets new generative AI performance and scale records in MLPerf Training v4.0** Using NVIDIA NeMo Framework and NVIDIA Hopper GPUs NVIDIA was able to scale to 11,616 H100 GPUs and achieve near-linear performance scaling on LLM pretraining.
*   **[2024/03/16] Accelerate your generative AI journey with NVIDIA NeMo Framework on GKE:** An end-to-end walkthrough to train generative AI models on the Google Kubernetes Engine (GKE) using the NVIDIA NeMo Framework is available.
*   **[2024/09/24] Accelerating Leaderboard-Topping ASR Models 10x with NVIDIA NeMo:** NVIDIA NeMo team released a number of inference optimizations for CTC, RNN-T, and TDT models that resulted in up to 10x inference speed-up.
*   **[2024/04/18] New Standard for Speech Recognition and Translation from the NVIDIA NeMo Canary Model:** The NeMo team just released Canary, a multilingual model that transcribes speech in English, Spanish, German, and French with punctuation and capitalization.
*   **[2024/04/18] Pushing the Boundaries of Speech Recognition with NVIDIA NeMo Parakeet ASR Models:** NVIDIA NeMo, an end-to-end platform for the development of multimodal generative AI models at scale anywhere—on any cloud and on-premises—released the Parakeet family of automatic speech recognition (ASR) models.
*   **[2024/04/18] Turbocharge ASR Accuracy and Speed with NVIDIA NeMo Parakeet-TDT:** NVIDIA NeMo, an end-to-end platform for developing multimodal generative AI models at scale anywhere—on any cloud and on-premises—recently released Parakeet-TDT.

## Get Started

*   **Quickstart Guide:** Explore using NeMo-Run to launch experiments.
*   **User Guide:** Dive deeper into NeMo 2.0 features and concepts.
*   **Recipes:** Find examples of large-scale runs with NeMo 2.0 and NeMo-Run.
*   **Feature Guide:** Learn about the main features of NeMo 2.0.
*   **Migration Guide:** Transition from NeMo 1.0 to 2.0.
*   **Pre-trained Models:** Access models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
*   **Tutorials:** Run tutorials on [Google Colab](https://colab.research.google.com) or with our [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Playbooks:** Learn to train NeMo models with the NeMo Framework Launcher.
*   **Example Scripts:** Access example scripts for advanced training and fine-tuning.

## Requirements:

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Installation

*   **Conda / Pip:** Install with native Pip into a virtual environment. (Recommended for ASR and TTS)
*   **NGC PyTorch Container:** Install from source into a highly optimized container.
*   **NGC NeMo Container:** Use a ready-to-go, high-performance container.

## Developer Documentation

See the documentation for the [latest branch](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).

## Support Matrix:

NeMo-Framework provides tiers of support based on OS / Platform and mode of installation. Please refer the following overview of support levels:

| OS / Platform              | Install from PyPi | Source into NGC container |
|----------------------------|-------------------|---------------------------|
| `linux` - `amd64/x84_64`   | Limited support   | Full support              |
| `linux` - `arm64`          | Limited support   | Limited support           |
| `darwin` - `amd64/x64_64`  | Deprecated        | Deprecated                |
| `darwin` - `arm64`         | Limited support   | Limited support           |
| `windows` - `amd64/x64_64` | No support yet    | No support yet            |
| `windows` - `arm64`        | No support yet    | No support yet            |

## Contribute

We welcome community contributions! Please refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Publications

Explore publications utilizing NeMo [here](https://nvidia.github.io/NeMo/publications/).

## Discussions

Find answers and engage with the community on the [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## License

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).