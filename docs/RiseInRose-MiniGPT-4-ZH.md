# MiniGPT-4: Revolutionizing Vision-Language Understanding with Advanced LLMs

**MiniGPT-4 empowers you to understand images by bridging the gap between vision and language using cutting-edge large language models.**  Find the original repository [here](https://github.com/RiseInRose/MiniGPT-4-ZH).

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny.

**Affiliation:** King Abdullah University of Science and Technology (KAUST)

## Key Features:

*   **Visual-Language Understanding:**  MiniGPT-4 excels at interpreting images and generating detailed, context-aware descriptions.
*   **Two-Stage Training:** Utilizing a two-stage training process, the model first aligns visual and language models and then fine-tunes for enhanced generation quality.
*   **Efficient Fine-tuning:**  The second stage fine-tuning process is highly computationally efficient.
*   **Open-Source & Accessible:** Leveraging open-source components like BLIP-2 and Vicuna, making it accessible for experimentation and application.
*   **Emerging Capabilities:**  MiniGPT-4 demonstrates advanced visual language abilities, similar to those seen in GPT-4.

## Quick Start:

### Online Demo

Interact with MiniGPT-4 directly to explore its capabilities:

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Find more examples on the [Project Page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

### Installation

**1.  Set up the Environment:**

    Clone the repository, create a Python environment, and activate it:

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    conda env create -f environment.yml
    conda activate minigpt4
    ```

**2.  Prepare Vicuna Weights:**

    Prepare Vicuna weights by following the instructions [here](PrepareVicuna.md).
    (Note: You may need to obtain the original LLaMA weights. Refer to the instructions in the original README for details.)

**3.  Prepare Pre-trained MiniGPT-4 Checkpoint:**

    Download the pre-trained checkpoint based on your Vicuna model (13B or 7B):
    *   [Checkpoint Aligned with Vicuna 13B](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
    *   [Checkpoint Aligned with Vicuna 7B](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

**4.  Run the Demo Locally:**

    Launch the demo on your machine:

    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    ```

## Training:

MiniGPT-4 is trained in two stages:

1.  **Stage 1 (Pretraining):** Align visual and language models using image-text pairs. Instructions for preparing the dataset can be found in `dataset/README_1_STAGE.md`.

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```
2.  **Stage 2 (Fine-tuning):**  Fine-tune the model using a high-quality image-text dataset.  Dataset preparation instructions are in `dataset/README_2_STAGE.md`.

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Acknowledgements:

*   **BLIP-2:**  MiniGPT-4's architecture is based on BLIP-2.
*   **Lavis:** The repository is built upon Lavis.
*   **Vicuna:** We are grateful for Vicuna's impressive language capabilities.

## Citation:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License:

This repository is licensed under the [BSD 3-Clause License](LICENSE.md). Code is based on [Lavis](https://github.com/salesforce/LAVIS), which is BSD 3-Clause licensed [here](LICENSE_Lavis.md).

## Community

*   [Join the domestic AI business application exchange group](https://mp.weixin.qq.com/s/XXXXXXXXXXXX)
*   [Knowledge Planet](https://xxxxxxxxxxx)

---