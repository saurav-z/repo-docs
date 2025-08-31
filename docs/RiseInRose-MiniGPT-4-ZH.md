# MiniGPT-4: Unleash Visual Language Understanding with Advanced LLMs

**Experience the power of MiniGPT-4, a groundbreaking model that bridges the gap between images and language using cutting-edge large language models.**  [Go to the original repository](https://github.com/RiseInRose/MiniGPT-4-ZH)

**Key Features:**

*   **Visual-Language Alignment:** MiniGPT-4 seamlessly connects visual information with text, allowing it to understand and generate text based on images.
*   **Enhanced Capabilities:**  Leverages the power of the Vicuna LLM to achieve impressive visual language understanding, leading to more reliable and useful outputs.
*   **Two-Stage Training:**  Employs a two-stage training process:
    *   **Stage 1 (Pretraining):**  Aligns vision and language models using a large dataset of image-text pairs.
    *   **Stage 2 (Fine-tuning):**  Refines the model with a smaller, high-quality dataset to significantly enhance generation quality and usability.
*   **User-Friendly:**  Offers a straightforward setup and intuitive use, even for those new to the field.
*   **Efficient:** Achieves impressive performance with reasonable resource requirements.

---

## Online Demo

Interact with MiniGPT-4 directly by uploading an image and see it describe and answer questions about your image!

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples on the [project page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

## News

*   A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! The demo's GPU memory consumption can be as low as 12GB.

---

## Introduction

*   MiniGPT-4 aligns a frozen visual encoder from BLIP-2 with the frozen LLM Vicuna using a projection layer.
*   MiniGPT-4 is trained through two stages.  The first is a pretraining stage (around 10 hours with 4 A100s) using approximately 5 million image-text pairs.
*   To improve usability, a novel method of creating high-quality image-text pairs through the model and ChatGPT itself is proposed. A high-quality dataset (3500 pairs) was built based on this method.
*   The second fine-tuning stage uses the dataset to improve the quality and reliability of the model. It is computationally efficient, only taking about 7 minutes on a single A100.
*   MiniGPT-4 showcases many emergent visual language capabilities similar to those demonstrated in GPT-4.

![overview](figs/overview.png)

---

## Getting Started

### Installation

**1.  Set up Code and Environment:**

*   Clone the repository:
    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    ```
*   Create and activate a Python environment:
    ```bash
    conda env create -f environment.yml
    conda activate minigpt4
    ```

**2. Prepare Vicuna Weights:**

*   (Option 1: Skip if you have the weights.  These may require manual downloading.  Consult the instructions for obtaining the LLaMA and Vicuna deltas)
*   (Option 2: Refer to the [original documentation](README_ENGLISH.md) for instructions on preparing the Vicuna weights and LLaMA base weights (including links and example commands), or using the [Hugging Face instructions](https://huggingface.co/transformers/model_doc/gpt2.html#transformers-gpt2-preprocessing-script) or other sources.)
*   (Additional Notes: Includes warnings about required memory, potential issues and solutions such as increasing swap space, and links to troubleshooting resources)

**3. Prepare MiniGPT-4 Checkpoints:**

*   Download the appropriate pre-trained checkpoint for your Vicuna model (13B or 7B).
    *   13B Checkpoint: [Download Link](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
    *   7B Checkpoint: [Download Link](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)
*   Set the path to the pre-trained checkpoint in the evaluation configuration file ( `eval_configs/minigpt4_eval.yaml#L10`).

### Run the Demo Locally

*   Run the demo:
    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    ```
*   Adjust configurations such as `low_resource` and search width for optimal performance based on your GPU memory.

---

### Training

MiniGPT-4 training involves two stages:

**1. Stage 1: Pretraining**

*   Train the model with image-text pairs from the Laion and CC datasets to align visual and language models.  See the documentation for data preparation details.
*   Run the training command:
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```
*   A Stage 1 checkpoint is available for download [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2. Stage 2: Fine-tuning**

*   Use a custom dataset in dialogue format to further align MiniGPT-4.  See the documentation for data preparation details.
*   Specify the Stage 1 checkpoint and output paths in `train_configs/minigpt4_stage2_finetune.yaml`.
*   Run the fine-tuning command:
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

---

## Acknowledgements

*   **BLIP2:** The model architecture is based on BLIP-2.
*   **Lavis:** This repository is built upon Lavis.
*   **Vicuna:**  Thanks to the incredible open-source Vicuna project.

If you use MiniGPT-4 in your research, please cite:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

---

## Community

*   [Join the AI Commercial Application Exchange Group (in Chinese)](#国内交流群) for updates and discussions.  Includes links to a public WeChat group and a knowledge-sharing platform.

---

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md).  The code is based on [Lavis](https://github.com/salesforce/LAVIS), licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).

---

## Thanks

*   Project forked from: [https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
*   Translation largely based on: [https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)