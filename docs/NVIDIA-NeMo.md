[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models

NVIDIA NeMo is a cloud-native framework designed for researchers and developers to efficiently create, customize, and deploy state-of-the-art generative AI models for various applications, including Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).  Learn more at the [original repo](https://github.com/NVIDIA/NeMo).

**Key Features:**

*   **Comprehensive Support:** Framework for LLMs, MMs, ASR, TTS, and CV.
*   **Scalability:** Designed for training on thousands of GPUs.
*   **Modular Design:**  Leverages PyTorch Lightning for flexibility and ease of use.
*   **Pre-trained Models:** Access a wide range of pre-trained models on Hugging Face Hub and NVIDIA NGC.
*   **Optimized Performance:** Integrates NVIDIA Transformer Engine and Megatron Core for efficient training.
*   **Alignment Techniques:** Supports SFT, PEFT, DPO, and RLHF for LLM fine-tuning.
*   **Deployment Ready:** Optimized for deployment with NVIDIA Riva and NeMo Microservices.
*   **Integration with Cosmos:** Supports training and customization of the NVIDIA Cosmos collection of world foundation models.

##  What's New in NeMo 2.0

NeMo 2.0 introduces significant improvements over its predecessor, including:

*   **Python-Based Configuration:** Enhanced flexibility and control through Python-based configuration.
*   **Modular Abstractions:** Simplified adaptation and experimentation with PyTorch Lightning's modular abstractions.
*   **Scalability:** Streamlined large-scale experiments with NeMo-Run.

## Getting Started with NeMo 2.0

*   Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples of using NeMo-Run.
*   For more information about NeMo 2.0, see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
*   [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) contains additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run.
*   For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
*   To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.

##  Cosmos Support

NVIDIA Cosmos allows for training and customizing the NVIDIA Cosmos collection of world foundation models.

*   NGC: [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos)
*   Hugging Face: [https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6)
*   Video datasets, refer to [NeMo Curator](https://developer.nvidia.com/nemo-curator).

## Training, Alignment, and Customization

*   Scalable training on thousands of GPUs.
*   Leverages cutting-edge distributed training techniques like Tensor Parallelism (TP), Pipeline Parallelism (PP), FSDP, MoE, and mixed precision training.
*   Uses NVIDIA Transformer Engine and NVIDIA Megatron Core for performance.
*   Supports advanced alignment techniques, including SFT, DPO, RLHF, and more.
*   Supports PEFT techniques such as LoRA, P-Tuning, Adapters, and IA3.

## Deployment and Optimization

*   Deploy and optimize LLMs and MMs with [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access).
*   Optimize ASR and TTS models for production with [NVIDIA Riva](https://developer.nvidia.com/riva).

## Key Features

*   [Large Language Models](nemo/collections/nlp/README.md)
*   [Multimodal](nemo/collections/multimodal/README.md)
*   [Automatic Speech Recognition](nemo/collections/asr/README.md)
*   [Text to Speech](nemo/collections/tts/README.md)
*   [Computer Vision](nemo/collections/vision/README.md)

##  Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (if you intend to do model training)

## Developer Documentation

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Install

The NeMo Framework can be installed in a variety of ways, depending on
your needs. Depending on the domain, you may find one of the following
installation methods more suitable.

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

-   Fully supported: Max performance and feature-completeness.
-   Limited supported: Used to explore NeMo.
-   No support yet: In development.
-   Deprecated: Support has reached end of life.

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

FAQ can be found on the NeMo [Discussions
board](https://github.com/NVIDIA/NeMo/discussions). You are welcome to
ask questions or start discussions on the board.

## Contribute to NeMo

We welcome community contributions! Please refer to
[CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md)
for the process.

## Publications

We provide an ever-growing list of
[publications](https://nvidia.github.io/NeMo/publications/) that utilize
the NeMo Framework.

To contribute an article to the collection, please submit a pull request
to the `gh-pages-src` branch of this repository. For detailed
information, please consult the README located at the [gh-pages-src
branch](https://github.com/NVIDIA/NeMo/tree/gh-pages-src#readme).

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