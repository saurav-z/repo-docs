[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy State-of-the-Art Generative AI Models

NVIDIA NeMo is a cloud-native framework that empowers researchers and developers to build, customize, and deploy cutting-edge generative AI models across various domains.  [View the original repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy advanced language models.
*   **Multimodal Models (MMs):** Develop AI models that can process and generate multiple types of data, such as text and images.
*   **Automatic Speech Recognition (ASR):** Build and optimize models for transcribing speech.
*   **Text-to-Speech (TTS):** Create high-quality speech synthesis models.
*   **Computer Vision (CV):** Develop and deploy computer vision models for various applications.
*   **Scalable Training:** Leverage distributed training techniques for efficient training on thousands of GPUs.
*   **Model Optimization:** Utilize NVIDIA Riva for inference optimization and deployment.
*   **Pre-trained Models:** Access a library of pre-trained models on Hugging Face Hub and NVIDIA NGC to accelerate your projects.
*   **NeMo 2.0:** Experience improved modularity, Python-based configuration, and enhanced scalability.

## What's New

*   **Hugging Face Model Support:**  NeMo now seamlessly integrates with Hugging Face models using AutoModel.
*   **Blackwell Support:** Improved performance benchmarks on GB200 & B200.
*   **Performance Tuning Guide:**  A comprehensive guide for performance tuning to achieve optimal throughput is available.
*   **New Model Support:** Support for latest community models - Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **Cosmos World Foundation Models Support:**  End-to-end NVIDIA Cosmos platform accelerates world model development for physical AI systems.

## Getting Started

### Installation

Choose the installation method that best suits your needs.

*   **Conda / Pip:**  Install the `nemo_toolkit` using pip or conda for general exploration.  This is recommended for ASR and TTS domains.
*   **NGC PyTorch container:** Install from source into a highly optimized container.
*   **NGC NeMo container:** For best performance, utilize a pre-built container with all dependencies installed and optimized.

### Detailed instructions

Detailed installation instructions are available in the [Install NeMo Framework](#install-nemo-framework) section below.

## Developer Resources

*   **Documentation:**  Comprehensive documentation is available at [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).
*   **Tutorials:**  Get started with tutorials that can be run on [Google Colab](https://colab.research.google.com) or with our [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Example Scripts:**  Find example scripts for multi-GPU/multi-node training in the [examples](https://github.com/NVIDIA/NeMo/tree/main/examples) directory.
*   **Publications:**  Explore publications that utilize the NeMo Framework at [publications](https://nvidia.github.io/NeMo/publications/).
*   **Discussions Board:**  Find answers to your questions or start discussions on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Contribute

We welcome contributions! See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Licensing

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).

## Install NeMo Framework

The NeMo Framework can be installed in a variety of ways, depending on
your needs. Depending on the domain, you may find one of the following
installation methods more suitable.

- [Conda / Pip](#conda--pip): Install NeMo-Framework with native Pip into a virtual environment.
  - Used to explore NeMo on any supported platform.
  - This is the recommended method for ASR and TTS domains.
  - Limited feature-completeness for other domains.
- [NGC PyTorch container](#ngc-pytorch-container): Install NeMo-Framework from source with feature-completeness into a highly optimized container.
  - For users that want to install from source in a highly optimized container.
- [NGC NeMo container](#ngc-nemo-container): Ready-to-go solution of NeMo-Framework
  - For users that seek highest performance.
  - Contains all dependencies installed and tested for performance and convergence.

### Support matrix

NeMo-Framework provides tiers of support based on OS / Platform and mode of installation. Please refer the following overview of support levels:

- Fully supported: Max performance and feature-completeness.
- Limited supported: Used to explore NeMo.
- No support yet: In development.
- Deprecated: Support has reached end of life.

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