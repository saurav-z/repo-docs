[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Accelerate Your Generative AI Journey with a Powerful, Cloud-Native Framework

[**Explore the NVIDIA NeMo Framework on GitHub**](https://github.com/NVIDIA/NeMo)

NVIDIA NeMo is a comprehensive, cloud-native framework empowering researchers and developers to build, customize, and deploy state-of-the-art generative AI models across various domains.

## Key Features

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy advanced LLMs.
*   **Multimodal Models (MMs):** Develop models that process and generate content across multiple modalities (text, images, video, etc.).
*   **Automatic Speech Recognition (ASR):** Create high-accuracy speech-to-text models.
*   **Text-to-Speech (TTS):** Build realistic and customizable speech synthesis systems.
*   **Computer Vision (CV):** Develop and deploy cutting-edge computer vision models.
*   **Modular and Scalable:** Built with PyTorch Lightning for flexibility and scalability across thousands of GPUs.
*   **Pre-trained Models and Recipes:** Access a wealth of pre-trained models and example scripts to accelerate development.
*   **Optimized for Performance:** Leverages NVIDIA Transformer Engine and Megatron Core for efficient training and deployment.
*   **Deployment with NVIDIA Riva:** Easily deploy and optimize ASR and TTS models using NVIDIA Riva.

## What's New

*   **Hugging Face Integration:** NeMo's AutoModel feature now provides broad support for Hugging Face models.
*   **Blackwell Support:** Experience optimized performance on GB200 & B200.
*   **Performance Tuning Guide:** Access a comprehensive guide for performance tuning to achieve optimal throughput.
*   **Latest Model Support:** Includes support for models like Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo 2.0 Release:** Enhanced modularity and ease of use, with a Python-based configuration and modular abstractions.  Refer to the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html) to get started.
*   **Cosmos Support:** Now supports training and customizing the NVIDIA Cosmos collection of world foundation models.

## Introduction

NVIDIA NeMo Framework is a scalable and cloud-native generative AI framework designed for researchers and PyTorch developers.  It facilitates the efficient creation, customization, and deployment of new generative AI models by leveraging existing code and pre-trained model checkpoints, and supports LLMs, MMs, ASR, TTS, and CV domains.

For technical documentation, please see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## LLMs and MMs Training, Alignment, and Customization

All NeMo models are trained with [Lightning](https://github.com/Lightning-AI/lightning). Training is automatically scalable to 1000s of GPUs. You can check the performance benchmarks using the latest NeMo Framework container [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

When applicable, NeMo models leverage cutting-edge distributed training techniques, incorporating [parallelism strategies](https://docs.nvidia.com/nemo-framework/user-guide/latest/modeloverview.html) to enable efficient training of very large models. These techniques include Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and Mixed Precision Training with BFloat16 and FP8, as well as others.

NeMo Transformer-based LLMs and MMs utilize [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) for FP8 training on NVIDIA Hopper GPUs, while leveraging [NVIDIA Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) for scaling Transformer model training.

NeMo LLMs can be aligned with state-of-the-art methods such as SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF). See [NVIDIA NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner) for more information.

In addition to supervised fine-tuning (SFT), NeMo also supports the latest parameter efficient fine-tuning (PEFT) techniques such as LoRA, P-Tuning, Adapters, and IA3. Refer to the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html) for the full list of supported models and techniques.

## LLMs and MMs Deployment and Optimization

NeMo LLMs and MMs can be deployed and optimized with [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access).

## Speech AI

NeMo ASR and TTS models can be optimized for inference and deployed for production use cases with [NVIDIA Riva](https://developer.nvidia.com/riva).

## Get Started with NeMo Framework

Getting started with NeMo Framework is easy. State-of-the-art pretrained NeMo models are freely available on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC). These models can be used to generate text or images, transcribe audio, and synthesize speech in just a few lines of code.

We have extensive [tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) that can be run on [Google Colab](https://colab.research.google.com) or with our [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo). We also have [playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) for users who want to train NeMo models with the NeMo Framework Launcher.

For advanced users who want to train NeMo models from scratch or fine-tune existing NeMo models, we have a full suite of [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) that support multi-GPU/multi-node training.

## Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (if you intend to do model training)

## Install NeMo Framework

The NeMo Framework can be installed in a variety of ways, depending on your needs. Depending on the domain, you may find one of the following installation methods more suitable.

*   [Conda / Pip](#conda--pip): Install NeMo-Framework with native Pip into a virtual environment.
    *   Used to explore NeMo on any supported platform.
    *   This is the recommended method for ASR and TTS domains.
    *   Limited feature-completeness for other domains.
*   [NGC PyTorch container](#ngc-pytorch-container): Install NeMo-Framework from source with feature-completeness into a highly optimized container.
    *   For users that want to install from source in a highly optimized container.
*   [NGC NeMo container](#ngc-nemo-container): Ready-to-go solution of NeMo-Framework
    *   For users that seek highest performance.
    *   Contains all dependencies installed and tested for performance and convergence.

### Support matrix

NeMo-Framework provides tiers of support based on OS / Platform and mode of installation. Please refer the following overview of support levels:

*   Fully supported: Max performance and feature-completeness.
*   Limited supported: Used to explore NeMo.
*   No support yet: In development.
*   Deprecated: Support has reached end of life.

Please refer to the following table for current support levels:

| OS / Platform              | Install from PyPi | Source into NGC container |
|----------------------------|-------------------|---------------------------|
| `linux` - `amd64/x84_64`   | Limited support   | Full support              |
| `linux` - `arm64`          | Limited support   | Limited support           |
| `darwin` - `amd64/x64_64`  | Deprecated        | Deprecated                |
| `darwin` - `arm64`         | Limited support   | Limited support           |
| `windows` - `amd64/x64_64` | No support yet    | No support yet            |
| `windows` - `arm64`        | No support yet    | No support yet            |

### Conda / Pip

Install NeMo in a fresh Conda environment:

```bash
conda create --name nemo python==3.10.12
conda activate nemo
```

#### Pick the right version

NeMo-Framework publishes pre-built wheels with each release.
To install nemo_toolkit from such a wheel, use the following installation method:

```bash
pip install "nemo_toolkit[all]"
```

If a more specific version is desired, we recommend a Pip-VCS install. From [NVIDIA/NeMo](github.com/NVIDIA/NeMo), fetch the commit, branch, or tag that you would like to install.  
To install nemo_toolkit from this Git reference `$REF`, use the following installation method:

```bash
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout @${REF:-'main'}
pip install '.[all]'
```

#### Install a specific Domain

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

### NGC PyTorch container

**NOTE: The following steps are supported beginning with 24.04 (NeMo-Toolkit 2.3.0)**

We recommended that you start with a base NVIDIA PyTorch container:
nvcr.io/nvidia/pytorch:25.01-py3.

If starting with a base NVIDIA PyTorch container, you must first launch
the container:

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

From [NVIDIA/NeMo](github.com/NVIDIA/NeMo), fetch the commit/branch/tag that you want to install.  
To install nemo_toolkit including all of its dependencies from this Git reference `$REF`, use the following installation method:

```bash
cd /opt
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout ${REF:-'main'}
bash docker/common/install_dep.sh --library all
pip install ".[all]"
```

## NGC NeMo container

NeMo containers are launched concurrently with NeMo version updates.
NeMo Framework now supports LLMs, MMs, ASR, and TTS in a single
consolidated Docker container. You can find additional information about
released containers on the [NeMo releases
page](https://github.com/NVIDIA/NeMo/releases).

To use a pre-built container, run the following code:

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

## Future Work

The NeMo Framework Launcher does not currently support ASR and TTS
training, but it will soon.

## Discussions Board

FAQ can be found on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions). You are welcome to ask questions or start discussions on the board.

## Contribute to NeMo

We welcome community contributions! Please refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for the process.

## Publications

We provide an ever-growing list of [publications](https://nvidia.github.io/NeMo/publications/) that utilize the NeMo Framework.

To contribute an article to the collection, please submit a pull request to the `gh-pages-src` branch of this repository. For detailed information, please consult the README located at the [gh-pages-src branch](https://github.com/NVIDIA/NeMo/tree/gh-pages-src#readme).

## Blogs

<!-- markdownlint-disable -->
<details open>
  <summary><b>Large Language Models and Multimodal Models</b></summary>
    <details>
      <summary>
        <a href="https://blogs.nvidia.com/blog/bria-builds-responsible-generative-ai-using-nemo-picasso/">
          Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso
        </a> (2024/03/06)
      </summary>
      Bria, a Tel Aviv startup at the forefront of visual generative AI for enterprises now leverages the NVIDIA NeMo Framework. 
      The Bria.ai platform uses reference implementations from the NeMo Multimodal collection, trained on NVIDIA Tensor Core GPUs, to enable high-throughput and low-latency image generation. 
      Bria has also adopted NVIDIA Picasso, a foundry for visual generative AI models, to run inference.
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/">
          New NVIDIA NeMo Framework Features and NVIDIA H200
        </a> (2023/12/06)
      </summary>
      NVIDIA NeMo Framework now includes several optimizations and enhancements, 
      including: 
      1) Fully Sharded Data Parallelism (FSDP) to improve the efficiency of training large-scale AI models, 
      2) Mix of Experts (MoE)-based LLM architectures with expert parallelism for efficient LLM training at scale, 
      3) Reinforcement Learning from Human Feedback (RLHF) with TensorRT-LLM for inference stage acceleration, and 
      4) up to 4.2x speedups for Llama 2 pre-training on NVIDIA H200 Tensor Core GPUs.
      <br><br>
      <a href="https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility">
      <img src="https://github.com/sbhavani/TransformerEngine/blob/main/docs/examples/H200-NeMo-performance.png" alt="H200-NeMo-performance" style="width: 600px;"></a>
      <br><br>
    </details>
    <details>
      <summary>
        <a href="https://blogs.nvidia.com/blog/nemo-amazon-titan/">
          NVIDIA now powers training for Amazon Titan Foundation models
        </a> (2023/11/28)
      </summary>
      NVIDIA NeMo Framework now empowers the Amazon Titan foundation models (FM) with efficient training of large language models (LLMs). 
      The Titan FMs form the basis of Amazon’s generative AI service, Amazon Bedrock. 
      The NeMo Framework provides a versatile framework for building, customizing, and running LLMs.
      <br><br>
    </details>
</details>
<!-- markdownlint-enable -->

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).