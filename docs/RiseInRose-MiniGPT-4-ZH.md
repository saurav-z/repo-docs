# MiniGPT-4: Unleashing Visual Language Understanding with Large Language Models

**Enhance your image understanding with MiniGPT-4, a powerful vision-language model that bridges the gap between images and text.**  For the original project, visit the [MiniGPT-4 GitHub repository](https://github.com/RiseInRose/MiniGPT-4-ZH).

*   **Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. (King Abdullah University of Science and Technology)

## Key Features

*   **Image-to-Text Generation:** Describe images with detailed and coherent text.
*   **Two-Stage Training:**  Utilizes a two-stage training process for robust performance.
*   **Improved Usability:** Enhanced generation quality and overall usability through fine-tuning.
*   **Efficient Fine-tuning:**  The second stage of fine-tuning is computationally efficient.
*   **Open Source:** Built upon open-source models like BLIP-2 and Vicuna.
*   **Online Demo:**  Interact directly with the model via a user-friendly online demo.
*   **Model Choices:** Compatible with both Vicuna-13B and Vicuna-7B models.

## Online Demo

Try out MiniGPT-4 by interacting with the online demo:
[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples and details on the [Project Page](https://minigpt-4.github.io).

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://minigpt-4.github.io)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](MiniGPT_4.pdf)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Vision-CAIR/MiniGPT-4)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News

*   A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! This reduces GPU memory consumption to as low as 12GB.

## Introduction

MiniGPT-4 combines a frozen visual encoder from BLIP-2 with the frozen LLM Vicuna, using a projection layer for alignment.

## Getting Started

### Installation

1.  **Clone the repository and create a Conda environment:**

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    conda env create -f environment.yml
    conda activate minigpt4
    ```

2.  **Prepare Vicuna Weights:**

    Follow the instructions in [PrepareVicuna.md](PrepareVicuna.md) to prepare the Vicuna weights.  Simplified instructions are provided below:

    *   Download the Vicuna delta weights from:  `https://huggingface.co/lmsys/vicuna-13b-delta-v1.1`
    *   Obtain the original LLaMA-13B weights (download links/methods are provided in the original README).
    *   Convert the LLaMA weights to Hugging Face format (using the `convert_llama_weights_to_hf.py` script and the FastChat library).
    *   Apply the delta weights to create the working Vicuna weights using the `fastchat.model.apply_delta` command.
    *   Specify the path to the Vicuna weights in the configuration file.

3.  **Prepare MiniGPT-4 Checkpoints:**

    Download the pre-trained checkpoints (links provided in the original README) and set the path in `eval_configs/minigpt4_eval.yaml`.

### Run the Demo Locally

Run the following command to start the demo:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

## Training

MiniGPT-4 training consists of two alignment stages.

1.  **Stage 1: Pre-training**
    *   Train on image-text pairs from Laion and CC datasets.
    *   Prepare the dataset using the instructions in `dataset/README_1_STAGE.md`.
    *   Run the pre-training command:
        ```bash
        torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
        ```
    *   A pre-trained MiniGPT-4 checkpoint from Stage 1 is available for download.

2.  **Stage 2: Fine-tuning**
    *   Fine-tune with a small, high-quality image-text dataset formatted in a conversational style.
    *   Prepare the dataset using the instructions in `dataset/README_2_STAGE.md`.
    *   Specify the path to the Stage 1 checkpoint in `train_configs/minigpt4_stage2_finetune.yaml`.
    *   Run the fine-tuning command:
        ```bash
        torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
        ```

## Acknowledgements

*   [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

## Citation

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This repository is licensed under the [BSD 3-Clause License](LICENSE.md).

Many code parts are based on [Lavis](https://github.com/salesforce/LAVIS), licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).