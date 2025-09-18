# MiniGPT-4: Unleashing Visual Language Understanding with Large Language Models

MiniGPT-4 is a groundbreaking model that enhances visual language understanding by connecting a frozen visual encoder from BLIP-2 with the powerful Vicuna large language model.  [Explore the original repository](https://github.com/RiseInRose/MiniGPT-4-ZH) for more details and to contribute.

**Key Features:**

*   **Visual-Language Alignment:** Integrates a frozen visual encoder with a frozen LLM (Vicuna) for robust image understanding.
*   **Two-Stage Training:** Employs a pre-training phase with 5 million image-text pairs followed by a fine-tuning phase on a high-quality, curated dataset.
*   **Efficient Fine-tuning:** Achieves significant performance improvements with a highly efficient fine-tuning phase.
*   **Enhanced Generative Capabilities:** Produces coherent and reliable text descriptions and responses related to images, similar to GPT-4.
*   **Open-Source & Accessible:** Offers an open-source model, weights, and demo for easy experimentation and community contribution.

## Online Demo

Interact with MiniGPT-4 directly to understand images!

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Find more examples on the [project page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

## News

*   A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available!  Demo GPU memory consumption can be as low as 12GB.

---

## Overview

*   MiniGPT-4 aligns a frozen visual encoder from BLIP-2 with the frozen LLM Vicuna using a projection layer.
*   Training involved two stages: a pre-training phase and a fine-tuning phase.
*   A novel approach was used to create a high-quality image-text dataset, resulting in improved generation capabilities.

![overview](figs/overview.png)

## Getting Started

### Installation

**1. Prepare Code and Environment**

Clone the repository, create a Python environment, and activate it:

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare Pre-trained Vicuna Weights**

Follow instructions in [PrepareVicuna.md](PrepareVicuna.md) to prepare the Vicuna weights. The provided file contains detailed steps for downloading and converting the LLaMA weights.

**3. Prepare Pre-trained MiniGPT-4 Checkpoint**

Download the pre-trained checkpoint based on the Vicuna model you are using.

*   **Checkpoint Aligned with Vicuna 13B:** [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
*   **Checkpoint Aligned with Vicuna 7B:** [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

Then, set the path to the pre-trained checkpoint in line 11 of the evaluation configuration file [minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L10).

### Launching the Demo Locally

Run the following command to try the demo locally:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

### Model Trimming

Instructions for model trimming are available within the original repository.

## Training

MiniGPT-4 training comprises two alignment stages. Detailed instructions for both stages are provided in the original repository.

**1. Stage 1: Pre-training**

Use image-text pairs from the Laion and CC datasets. Instructions for preparing the dataset are in [dataset/README_1_STAGE.md](dataset/README_1_STAGE.md). Run the following command to begin stage 1 training:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

**2. Stage 2: Fine-tuning**

Use a curated, high-quality image-text dataset.  See [dataset/README_2_STAGE.md](dataset/README_2_STAGE.md) for dataset preparation.  Specify the checkpoint path from stage 1 in [train_configs/minigpt4_stage2_finetune.yaml](train_configs/minigpt4_stage2_finetune.yaml) and run:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Acknowledgements

*   [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [LAVIS](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

## Citation

If you use MiniGPT-4 in your research, please cite it as follows:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This repository is licensed under the [BSD 3-Clause License](LICENSE.md).  Many code components are based on [Lavis](https://github.com/salesforce/LAVIS) with its [BSD 3-Clause License](LICENSE_Lavis.md).