# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models at Scale

**NVIDIA NeMo is a powerful, cloud-native framework that simplifies the creation, customization, and deployment of generative AI models across various domains.** ([Back to original repo](https://github.com/NVIDIA/NeMo))

[![Project Status: Active](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Features

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy state-of-the-art LLMs.
*   **Multimodal Models (MMs):** Develop models that combine text, images, and more.
*   **Automatic Speech Recognition (ASR):** Build and optimize ASR models.
*   **Text-to-Speech (TTS):** Create high-quality speech synthesis models.
*   **Computer Vision (CV):** Implement and customize computer vision tasks.
*   **Modular Design:** Leverage PyTorch Lightning's modular approach for easy experimentation.
*   **Scalability:** Train models efficiently across thousands of GPUs using NeMo-Run.
*   **Parameter-Efficient Fine-tuning (PEFT):** Support for techniques like LoRA, and Adapters.

## What's New in NeMo

### Pretrain and finetune :hugs:Hugging Face models via AutoModel
  Nemo Framework's latest feature AutoModel enables broad support for :hugs:Hugging Face models, with 25.04 focusing on

  
- <a href=https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#automodelforcausallm>AutoModelForCausalLM<a> in the <a href="https://huggingface.co/models?pipeline_tag=text-generation&sort=trending">Text Generation<a> category
- <a href=https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForImageTextToText>AutoModelForImageTextToText<a> in the <a href="https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending">Image-Text-to-Text<a> category

More Details in Blog: <a href=https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework>Run Hugging Face Models Instantly with Day-0 Support from NVIDIA NeMo Framework<a>. Future releases will enable support for more model families such as Video Generation models.(2025-05-19)

### Training on Blackwell using Nemo
  NeMo Framework has added Blackwell support, with <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html>performance benchmarks on GB200 & B200<a>. More optimizations to come in the upcoming releases.(2025-05-19)

### Training Performance on GPU Tuning Guide
  NeMo Framework has published <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html>a comprehensive guide for performance tuning to achieve optimal throughput<a>! (2025-05-19)

### New Models Support
  NeMo Framework has added support for latest community models - <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html>Llama 4<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/vision/diffusionmodels/flux.html>Flux<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama_nemotron.html>Llama Nemotron<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/hyena.html#>Hyena & Evo2<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/qwen2vl.html>Qwen2-VL<a>, <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/qwen2.html>Qwen2.5<a>, Gemma3, Qwen3-30B&32B.(2025-05-19)

### NeMo Framework 2.0
  We've released NeMo 2.0, an update on the NeMo Framework which prioritizes modularity and ease-of-use. Please refer to the <a href=https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html>NeMo Framework User Guide</a> to get started.

### New Cosmos World Foundation Models Support
  <details> 
    <summary> <a href="https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform">Advancing Physical AI with NVIDIA Cosmos World Foundation Model Platform </a> (2025-01-09) 
    </summary> 
      The end-to-end NVIDIA Cosmos platform accelerates world model development for physical AI systems. Built on CUDA, Cosmos combines state-of-the-art world foundation models, video tokenizers, and AI-accelerated data processing pipelines. Developers can accelerate world model development by fine-tuning Cosmos world foundation models or building new ones from the ground up. These models create realistic synthetic videos of environments and interactions, providing a scalable foundation for training complex systems, from simulating humanoid robots performing advanced actions to developing end-to-end autonomous driving models. 
      <br><br>
  </details>
  <details>
    <summary>
      <a href="https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities/">
        Accelerate Custom Video Foundation Model Pipelines with New NVIDIA NeMo Framework Capabilities
      </a> (2025-01-07)
    </summary>
      The NeMo Framework now supports training and customizing the <a href="https://github.com/NVIDIA/Cosmos">NVIDIA Cosmos</a> collection of world foundation models. Cosmos leverages advanced text-to-world generation techniques to create fluid, coherent video content from natural language prompts.
      <br><br>
      You can also now accelerate your video processing step using the <a href="https://developer.nvidia.com/nemo-curator-video-processing-early-access">NeMo Curator</a> library, which provides optimized video processing and captioning features that can deliver up to 89x faster video processing when compared to an unoptimized CPU pipeline.
    <br><br>
  </details>

### Large Language Models and Multimodal Models
  <details>
    <summary>
      <a href="https://developer.nvidia.com/blog/state-of-the-art-multimodal-generative-ai-model-development-with-nvidia-nemo/">
        State-of-the-Art Multimodal Generative AI Model Development with NVIDIA NeMo
      </a> (2024-11-06)
    </summary>
      NVIDIA recently announced significant enhancements to the NeMo platform, focusing on multimodal generative AI models. The update includes NeMo Curator and the Cosmos tokenizer, which streamline the data curation process and enhance the quality of visual data. These tools are designed to handle large-scale data efficiently, making it easier to develop high-quality AI models for various applications, including robotics and autonomous driving. The Cosmos tokenizers, in particular, efficiently map visual data into compact, semantic tokens, which is crucial for training large-scale generative models. The tokenizer is available now on the <a href=http://github.com/NVIDIA/cosmos-tokenizer/NVIDIA/cosmos-tokenizer>NVIDIA/cosmos-tokenizer</a> GitHub repo and on <a href=https://huggingface.co/nvidia/Cosmos-Tokenizer-CV8x8x8>Hugging Face</a>.
    <br><br>
  </details>
  <details>
    <summary>
      <a href="https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama/index.html#new-llama-3-1-support for more information/">
      New Llama 3.1 Support
      </a> (2024-07-23)
    </summary>
      The NeMo Framework now supports training and customizing the Llama 3.1 collection of LLMs from Meta.
    <br><br>
  </details>
  <details>
    <summary>
      <a href="https://aws.amazon.com/blogs/machine-learning/accelerate-your-generative-ai-distributed-training-workloads-with-the-nvidia-nemo-framework-on-amazon-eks/">
        Accelerate your Generative AI Distributed Training Workloads with the NVIDIA NeMo Framework on Amazon EKS
      </a> (2024-07-16)
    </summary>
    NVIDIA NeMo Framework now runs distributed training workloads on an Amazon Elastic Kubernetes Service (Amazon EKS) cluster. For step-by-step instructions on creating an EKS cluster and running distributed training workloads with NeMo, see the GitHub repository <a href="https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/2.nemo-launcher/EKS/"> here.</a>
    <br><br>
  </details>
  <details>
    <summary>
      <a href="https://developer.nvidia.com/blog/nvidia-nemo-accelerates-llm-innovation-with-hybrid-state-space-model-support/">
        NVIDIA NeMo Accelerates LLM Innovation with Hybrid State Space Model Support
      </a> (2024/06/17)
    </summary>
    NVIDIA NeMo and Megatron Core now support pre-training and fine-tuning of state space models (SSMs). NeMo also supports training models based on the Griffin architecture as described by Google DeepMind. 
    <br><br>
  </details>
    <details>
    <summary>
      <a href="https://huggingface.co/models?sort=trending&search=nvidia%2Fnemotron-4-340B">
        NVIDIA releases 340B base, instruct, and reward models pretrained on a total of 9T tokens.
      </a> (2024-06-18)
    </summary>
    See documentation and tutorials for SFT, PEFT, and PTQ with 
    <a href="https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/nemotron/index.html">
      Nemotron 340B 
    </a>
    in the NeMo Framework User Guide.
    <br><br>
  </details>
  <details>
    <summary>
      <a href="https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/">
        NVIDIA sets new generative AI performance and scale records in MLPerf Training v4.0
      </a> (2024/06/12)
    </summary>
    Using NVIDIA NeMo Framework and NVIDIA Hopper GPUs NVIDIA was able to scale to 11,616 H100 GPUs and achieve near-linear performance scaling on LLM pretraining. 
    NVIDIA also achieved the highest LLM fine-tuning performance and raised the bar for text-to-image training.
    <br><br>
  </details>
  <details>
    <summary>
      <a href="https://cloud.google.com/blog/products/compute/gke-and-nvidia-nemo-framework-to-train-generative-ai-models">
        Accelerate your generative AI journey with NVIDIA NeMo Framework on GKE
      </a> (2024/03/16)
    </summary>
    An end-to-end walkthrough to train generative AI models on the Google Kubernetes Engine (GKE) using the NVIDIA NeMo Framework is available at https://github.com/GoogleCloudPlatform/nvidia-nemo-on-gke. 
    The walkthrough includes detailed instructions on how to set up a Google Cloud Project and pre-train a GPT model using the NeMo Framework.
    <br><br>
  </details>

### Speech Recognition
  <details>
    <summary>
      <a href="https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/">
        Accelerating Leaderboard-Topping ASR Models 10x with NVIDIA NeMo
      </a> (2024/09/24)
    </summary>
    NVIDIA NeMo team released a number of inference optimizations for CTC, RNN-T, and TDT models that resulted in up to 10x inference speed-up. 
    These models now exceed an inverse real-time factor (RTFx) of 2,000, with some reaching RTFx of even 6,000.
    <br><br>
  </details>
    <details>
    <summary>
      <a href="https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/">
        New Standard for Speech Recognition and Translation from the NVIDIA NeMo Canary Model
      </a> (2024/04/18)
    </summary>
    The NeMo team just released Canary, a multilingual model that transcribes speech in English, Spanish, German, and French with punctuation and capitalization. 
    Canary also provides bi-directional translation, between English and the three other supported languages.
    <br><br>
  </details>
    <details>
    <summary>
      <a href="https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/">
        Pushing the Boundaries of Speech Recognition with NVIDIA NeMo Parakeet ASR Models
      </a> (2024/04/18)
    </summary>
    NVIDIA NeMo, an end-to-end platform for the development of multimodal generative AI models at scale anywhere—on any cloud and on-premises—released the Parakeet family of automatic speech recognition (ASR) models. 
    These state-of-the-art ASR models, developed in collaboration with Suno.ai, transcribe spoken English with exceptional accuracy.
    <br><br>
  </details>
  <details>
    <summary>
      <a href="https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt/">
        Turbocharge ASR Accuracy and Speed with NVIDIA NeMo Parakeet-TDT
      </a> (2024/04/18)
    </summary>
    NVIDIA NeMo, an end-to-end platform for developing multimodal generative AI models at scale anywhere—on any cloud and on-premises—recently released Parakeet-TDT. 
    This new addition to the  NeMo ASR Parakeet model family boasts better accuracy and 64% greater speed over the previously best model, Parakeet-RNNT-1.1B.
    <br><br>
  </details>

## Introduction

NVIDIA NeMo Framework is a scalable and cloud-native generative AI
framework built for researchers and PyTorch developers working on Large
Language Models (LLMs), Multimodal Models (MMs), Automatic Speech
Recognition (ASR), Text to Speech (TTS), and Computer Vision (CV)
domains. It is designed to help you efficiently create, customize, and
deploy new generative AI models by leveraging existing code and
pre-trained model checkpoints.

For technical documentation, please see the [NeMo Framework User
Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## What's New in NeMo 2.0

NVIDIA NeMo 2.0 introduces several significant improvements over its predecessor, NeMo 1.0, enhancing flexibility, performance, and scalability.

- **Python-Based Configuration** - NeMo 2.0 transitions from YAML files to a Python-based configuration, providing more flexibility and control. This shift makes it easier to extend and customize configurations programmatically.

- **Modular Abstractions** - By adopting PyTorch Lightning’s modular abstractions, NeMo 2.0 simplifies adaptation and experimentation. This modular approach allows developers to more easily modify and experiment with different components of their models.

- **Scalability** - NeMo 2.0 seamlessly scaling large-scale experiments across thousands of GPUs using [NeMo-Run](https://github.com/NVIDIA/NeMo-Run), a powerful tool designed to streamline the configuration, execution, and management of machine learning experiments across computing environments.

Overall, these enhancements make NeMo 2.0 a powerful, scalable, and user-friendly framework for AI model development.

> [!IMPORTANT]  
> NeMo 2.0 is currently supported by the LLM (large language model) and VLM (vision language model) collections.

### Get Started with NeMo 2.0

- Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples of using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster.
- For more information about NeMo 2.0, see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
- [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) contains additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run.
- For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
- To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.

### Get Started with Cosmos

NeMo Curator and NeMo Framework support video curation and post-training of the Cosmos World Foundation Models, which are open and available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos) and [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6). For more information on video datasets, refer to [NeMo Curator](https://developer.nvidia.com/nemo-curator). To post-train World Foundation Models using the NeMo Framework for your custom physical AI tasks, see the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## LLMs and MMs Training, Alignment, and Customization

All NeMo models are trained with
[Lightning](https://github.com/Lightning-AI/lightning). Training is
automatically scalable to 1000s of GPUs. You can check the performance benchmarks using the
latest NeMo Framework container [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

When applicable, NeMo models leverage cutting-edge distributed training
techniques, incorporating [parallelism
strategies](https://docs.nvidia.com/nemo-framework/user-guide/latest/modeloverview.html)
to enable efficient training of very large models. These techniques
include Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully
Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and Mixed
Precision Training with BFloat16 and FP8, as well as others.

NeMo Transformer-based LLMs and MMs utilize [NVIDIA Transformer
Engine](https://github.com/NVIDIA/TransformerEngine) for FP8 training on
NVIDIA Hopper GPUs, while leveraging [NVIDIA Megatron
Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) for
scaling Transformer model training.

NeMo LLMs can be aligned with state-of-the-art methods such as SteerLM,
Direct Preference Optimization (DPO), and Reinforcement Learning from
Human Feedback (RLHF). See [NVIDIA NeMo
Aligner](https://github.com/NVIDIA/NeMo-Aligner) for more information.

In addition to supervised fine-tuning (SFT), NeMo also supports the
latest parameter efficient fine-tuning (PEFT) techniques such as LoRA,
P-Tuning, Adapters, and IA3. Refer to the [NeMo Framework User
Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html)
for the full list of supported models and techniques.

## LLMs and MMs Deployment and Optimization

NeMo LLMs and MMs can be deployed and optimized with [NVIDIA NeMo
Microservices](https://developer.nvidia.com/nemo-microservices-early-access).

## Speech AI

NeMo ASR and TTS models can be optimized for inference and deployed for
production use cases with [NVIDIA Riva](https://developer.nvidia.com/riva).

## NeMo Framework Launcher

> [!IMPORTANT]  
> NeMo Framework Launcher is compatible with NeMo version 1.0 only. [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) is recommended for launching experiments using NeMo 2.0.

[NeMo Framework
Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) is a
cloud-native tool that streamlines the NeMo Framework experience. It is
used for launching end-to-end NeMo Framework training jobs on CSPs and
Slurm clusters.

The NeMo Framework Launcher includes extensive recipes, scripts,
utilities, and documentation for training NeMo LLMs. It also includes
the NeMo Framework [Autoconfigurator](https://github.com/NVIDIA/NeMo-Megatron-Launcher#53-using-autoconfigurator-to-find-the-optimal-configuration),
which is designed to find the optimal model parallel configuration for
training on a specific cluster.

To get started quickly with the NeMo Framework Launcher, please see the
[NeMo Framework
Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).
The NeMo Framework Launcher does not currently support ASR and TTS
training, but it will soon.

## Get Started with NeMo Framework

Getting started with NeMo Framework is easy. State-of-the-art pretrained
NeMo models are freely available on [Hugging Face
Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia)
and [NVIDIA
NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
These models can be used to generate text or images, transcribe audio,
and synthesize speech in just a few lines of code.

We have extensive
[tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html)
that can be run on [Google Colab](https://colab.research.google.com) or
with our [NGC NeMo Framework
Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
We also have
[playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)
for users who want to train NeMo models with the NeMo Framework
Launcher.

For advanced users who want to train NeMo models from scratch or
fine-tune existing NeMo models, we have a full suite of [example
scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) that support
multi-GPU/multi-node training.

## Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (if you intend to do model training)

## Installation

Choose the installation method that best suits your needs:

*   **[Conda / Pip](#conda--pip):**  Install NeMo with Pip into a virtual environment for general exploration and use.  (Recommended for ASR and TTS domains)
*   **[NGC PyTorch Container](#ngc-pytorch-container):** Install NeMo from source within an optimized container.
*   **[NGC NeMo Container](#ngc-nemo-container):** Use a pre-built, optimized container for maximum performance and ease of use.

### Conda / Pip

1.  Create and activate a Conda environment:
    ```bash
    conda create --name nemo python==3.10.12
    conda activate nemo
    ```
2.  Install `nemo_toolkit` using one of the following methods:

    *   **Install from pre-built wheels (Recommended):**
        ```bash
        pip install "nemo_toolkit[all]"
        ```

    *   **Install from a specific Git reference:**
        ```bash
        git clone https://github.com/NVIDIA/NeMo
        cd NeMo
        git checkout @${REF:-'main'}  # Replace 'main' with your desired branch/tag/commit
        pip install '.[all]'
        ```

    *   **Install a specific domain:**  After installing `nemo_toolkit`, install domain-specific dependencies:
        ```bash
        pip install nemo_toolkit['asr']
        pip install nemo_toolkit['nlp']
        pip install nemo_toolkit['tts']
        pip install nemo_toolkit['vision']
        pip install nemo_toolkit['multimodal']
        ```

### NGC PyTorch Container

1.  Launch the base NVIDIA PyTorch container (e.g., `nvcr.io/nvidia/pytorch:25.01-py3`):

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

2.  Inside the container:

    ```bash
    cd /opt
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout ${REF:-'main'}  # Replace 'main' with your desired branch/tag/commit
    bash docker/common/install_dep.sh --library all
    pip install ".[all]"
    ```

### NGC NeMo Container

Run the pre-built NeMo container:

```bash
docker run \
  --gpus all \
  -it \
  --rm \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'} # Check Releases for latest tag
```

## Developer Documentation

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Contribute

Contribute to NeMo by following the guidelines in [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).

## Discussions Board

Find answers to frequently asked questions and engage in discussions on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Publications

Explore a curated list of [publications](https://nvidia.github.io/NeMo/publications/) that utilize the NeMo Framework.  Contribute an article by submitting a pull request to the `gh-pages-src` branch.

## Blogs

*   [Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso](https://blogs.nvidia.com/blog/bria-builds-responsible-generative-ai-using-nemo-picasso/) (2024/03/06)
*   [New NVIDIA NeMo Framework Features and NVIDIA H200](https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/) (2023/12/06)
*   [NVIDIA now powers training for Amazon Titan Foundation models](https://blogs.nvidia.com/blog/nemo-amazon-titan/) (2023/11/28)

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).