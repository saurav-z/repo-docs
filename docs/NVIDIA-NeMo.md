[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a powerful, cloud-native framework designed to simplify the development and deployment of state-of-the-art generative AI models.  ([See the original repository](https://github.com/NVIDIA/NeMo)).

## Key Features

*   **Large Language Models (LLMs):** Efficiently train, fine-tune, and deploy LLMs.
*   **Multimodal Models (MMs):** Develop models that combine text, images, and more.
*   **Automatic Speech Recognition (ASR):** Build high-accuracy speech recognition systems.
*   **Text-to-Speech (TTS):** Create realistic and natural-sounding speech synthesis.
*   **Computer Vision (CV):** Utilize pre-trained models and develop custom vision applications.

## Latest Updates

*   **[May 2024]:** New Blackwell support and performance benchmarks, plus a comprehensive performance tuning guide.
*   **[May 2024]:** Support for the latest community models like Llama 4, Flux, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B, and others.
*   **[May 2024]:** Run Hugging Face models instantly with day-0 support via AutoModel for Text Generation and Image-Text-to-Text.
*   **[Jan 2024]:** Added support for the NVIDIA Cosmos collection of world foundation models and the new NeMo Curator library.

### **NVIDIA NeMo 2.0**

NVIDIA NeMo 2.0 emphasizes modularity and ease-of-use with Python-based configurations and PyTorch Lightning’s modular abstractions.  Refer to the  [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html) for more details.

## Getting Started

Leverage pre-trained models from [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) or [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC) to generate text, images, transcribe audio, and synthesize speech.

### Guides & Tutorials

*   [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for NeMo 2.0 experiments.
*   [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) for launching large-scale runs.
*   [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide) for main features of NeMo 2.0.
*   [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) to transition from NeMo 1.0 to 2.0.
*   [Tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) run on Google Colab.
*   [Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) for training with NeMo Framework Launcher.
*   [Example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) for advanced users.

## Installation

Choose your preferred method:

*   **Conda / Pip:**  For exploring NeMo on any supported platform, especially recommended for ASR and TTS.
*   **NGC PyTorch container:** For users wanting to install from source into a highly optimized container.
*   **NGC NeMo container:**  A ready-to-go solution for peak performance.

**Support Matrix**

| OS / Platform              | Install from PyPi | Source into NGC container |
|----------------------------|-------------------|---------------------------|
| `linux` - `amd64/x84_64`   | Limited support   | Full support              |
| `linux` - `arm64`          | Limited support   | Limited support           |
| `darwin` - `amd64/x64_64`  | Deprecated        | Deprecated                |
| `darwin` - `arm64`         | Limited support   | Limited support           |
| `windows` - `amd64/x64_64` | No support yet    | No support yet            |
| `windows` - `arm64`        | No support yet    | No support yet            |

### Conda / Pip Installation

1.  Create a Conda environment:

    ```bash
    conda create --name nemo python==3.10.12
    conda activate nemo
    ```

2.  Install using pre-built wheels:

    ```bash
    pip install "nemo_toolkit[all]"
    ```

3.  Install from a specific Git reference:

    ```bash
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout @${REF:-'main'}
    pip install '.[all]'
    ```

4.  Install specific domains (after installing `nemo_toolkit`):

    ```bash
    pip install nemo_toolkit['all']
    pip install nemo_toolkit['asr']
    pip install nemo_toolkit['nlp']
    pip install nemo_toolkit['tts']
    pip install nemo_toolkit['vision']
    pip install nemo_toolkit['multimodal']
    ```

### NGC PyTorch container

1.  Launch a base NVIDIA PyTorch container:

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

2.  Install using a specific Git reference:

    ```bash
    cd /opt
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout ${REF:-'main'}
    bash docker/common/install_dep.sh --library all
    pip install ".[all]"
    ```

### NGC NeMo container

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

## Advanced Capabilities

*   **Training and Fine-tuning:** Supports training LLMs and MMs with PyTorch Lightning, scales to thousands of GPUs, and uses techniques like TP, PP, FSDP, MoE, and Mixed Precision Training.
*   **NVIDIA Transformer Engine:** Utilizes FP8 training on NVIDIA Hopper GPUs.
*   **NVIDIA Megatron Core:** Leverages for scaling Transformer model training.
*   **Alignment and PEFT:** Supports state-of-the-art methods such as SteerLM, DPO, RLHF, LoRA, P-Tuning, Adapters, and IA3.
*   **Deployment and Optimization:** Deploy and optimize with NVIDIA NeMo Microservices and Riva.

## Community & Resources

*   **Discussions Board:** Find answers and engage in discussions on the [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions).
*   **Contribute:**  Contribute to NeMo; refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).
*   **Publications:**  Explore research using NeMo: [Publications](https://nvidia.github.io/NeMo/publications/).
*   **Blogs:**  Stay informed with the latest news: [Blogs](blogs/).

## Licenses

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).