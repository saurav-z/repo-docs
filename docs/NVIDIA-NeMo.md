[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models at Scale

NVIDIA NeMo is a flexible, cloud-native framework empowering researchers and developers to build, train, and deploy state-of-the-art generative AI models for various applications. **[View the original repo](https://github.com/NVIDIA/NeMo) for more details.**

## Key Features

*   **Large Language Models (LLMs):** Train and customize powerful LLMs.
*   **Multimodal Models (MMs):** Develop AI models that process multiple data types.
*   **Automatic Speech Recognition (ASR):** Build accurate speech recognition systems.
*   **Text-to-Speech (TTS):** Generate natural-sounding speech.
*   **Computer Vision (CV):** Create cutting-edge computer vision models.
*   **Scalability:** Train models efficiently across multiple GPUs with NeMo-Run.
*   **Modular Design:** Utilize modular abstractions for easier experimentation.
*   **Pre-trained Models:** Leverage readily available models on Hugging Face Hub and NVIDIA NGC.
*   **Integration:** Deploy and optimize models with NVIDIA Riva and NeMo Microservices.

## What's New - Stay Updated

*   **Latest News:** Access the latest features, model support and performance improvements from NVIDIA NeMo.

    *   **Pretrain and finetune Hugging Face models via AutoModel** (2025-05-19)
    *   **Training on Blackwell using Nemo** (2025-05-19)
    *   **Training Performance on GPU Tuning Guide** (2025-05-19)
    *   **New Models Support** (2025-05-19)
    *   **NeMo Framework 2.0** (2024-07-23)
    *   **New Cosmos World Foundation Models Support** (2025-01-09)

*   **LLMs and MMs Training, Alignment, and Customization:** Optimize your workflow.
*   **LLMs and MMs Deployment and Optimization:** Take advantage of NVIDIA's innovative technology.
*   **Speech AI:** Incorporate new ASR and TTS models.
*   **NeMo Framework Launcher:** Access a cloud-native tool to streamline NeMo Framework.

## Getting Started

NeMo offers several ways to get started:

*   **Pre-trained Models:** Utilize models from [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
*   **Tutorials:** Follow comprehensive tutorials on [Google Colab](https://colab.research.google.com) or with our [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Playbooks:** Train NeMo models with the NeMo Framework Launcher.
*   **Example Scripts:** Train models from scratch or fine-tune existing NeMo models.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## Installation

Choose the installation method that best suits your needs:

*   **Conda / Pip:** Explore NeMo.
*   **NGC PyTorch Container:** Install from source.
*   **NGC NeMo Container:** Ready-to-go solution.

### Support matrix

NeMo-Framework provides tiers of support based on OS / Platform and mode of installation. Please refer the following overview of support levels:

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

## Developer Documentation

Access detailed documentation for the latest and stable releases:

*   **Latest:** [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Stable:** [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)

## Resources

*   **Discussions Board:** Find answers and ask questions on the [Discussions board](https://github.com/NVIDIA/NeMo/discussions).
*   **Contribute:** Learn how to contribute in [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).
*   **Publications:** Explore articles using NeMo: [Publications](https://nvidia.github.io/NeMo/publications/).
*   **Blogs:** Stay up-to-date with the latest announcements: [Blogs](https://blogs.nvidia.com/).

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).