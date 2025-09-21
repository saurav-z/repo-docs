# MiniGPT-4: Unlock Visual Language Understanding with Advanced LLMs

**MiniGPT-4, developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny, bridges the gap between vision and language, enabling insightful image understanding.**  Check out the [original repo](https://github.com/RiseInRose/MiniGPT-4-ZH) for the full source code and details.

**Key Features:**

*   **Enhanced Visual Language Understanding:**  Leverages advanced Large Language Models (LLMs) for superior image comprehension.
*   **Two-Stage Training:**  Employs a two-stage training process, starting with pre-training on image-text pairs and followed by fine-tuning on a high-quality dataset.
*   **Simplified Deployment:**  Offers a streamlined process for setting up and running the model, including pre-trained weights and Colab notebooks.
*   **Dialogue-Based Interaction:**  Features dialogue-formatted training to significantly improve the generation reliability and overall usability.
*   **Open Source:**  Built upon open-source technologies like BLIP-2, Lavis, and Vicuna.

## Online Demo

Interact with MiniGPT-4 directly by uploading an image to learn more about it.

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore additional examples on the [project page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

## News

A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! The demo's GPU memory consumption can be as low as 12GB.

---

## Introduction

-   MiniGPT-4 aligns the frozen visual encoder from BLIP-2 with the frozen LLM Vicuna using a projection layer.
-   The model is trained in two stages.
-   We propose a new method to create high-quality image-text pairs using the model and ChatGPT itself. Based on this, we have created a small but high-quality dataset.
-   The second fine-tuning stage uses this dataset on a dialogue template to significantly improve its generation reliability and overall usability.
-   MiniGPT-4 is capable of producing many of the emerging visual language capabilities demonstrated in GPT-4.

![overview](figs/overview.png)

## Getting Started: Installation

**1.  Prepare the Code and Environment**

Clone the repository, create a Python environment, and activate it using:

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2.  Prepare Pre-trained Vicuna Weights**

Refer to the instructions [here](PrepareVicuna.md) to prepare the Vicuna weights.

### Prepare Vicuna Weights - Summary

1.  Download Vicuna delta weights (e.g., `vicuna-13b-delta-v1.1`)
2.  Get the original LLaMA-13B weights (download or use a torrent)
3.  Convert weights to Hugging Face format using `convert_llama_weights_to_hf.py`
4.  Apply delta weights using `apply_delta` script.

**3. Prepare Pre-trained MiniGPT-4 Checkpoint**

Download a pre-trained checkpoint based on your Vicuna model (13B or 7B).

*   [Checkpoint Aligned with Vicuna 13B](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
*   [Checkpoint Aligned with Vicuna 7B](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

## Run Demo Locally

Run the demo on your local machine:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

## Model Trimming

*   Model trimming can impact accuracy. Use with caution.

## Training

MiniGPT-4 training involves two alignment stages.

**1. Stage 1: Pre-training**

Train the model on image-text pairs from the Laion and CC datasets.  See `dataset/README_1_STAGE.md` for dataset preparation.

Run to start stage 1 training:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

[Stage 1 Checkpoint](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link)

**2. Stage 2: Fine-tuning**

Use a high-quality image-text dataset for dialogue-based alignment. See `dataset/README_2_STAGE.md` for dataset preparation.

Run to start stage 2 fine-tuning:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Acknowledgements

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

Cite MiniGPT-4:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## Community & Resources

*   Join the community for updates and support (links to WeChat groups and knowledge base).

## License

This repository uses the [BSD 3-Clause License](LICENSE.md).  Based on [Lavis](https://github.com/salesforce/LAVIS) using the [BSD 3-Clause License](LICENSE_Lavis.md).