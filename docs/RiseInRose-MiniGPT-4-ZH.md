# MiniGPT-4: Unleash Visual Language Understanding with Advanced LLMs

**MiniGPT-4 empowers you to converse with images, bridging the gap between vision and language.** Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology.  [Find the original repo here](https://github.com/RiseInRose/MiniGPT-4-ZH).

## Key Features

*   **Image-to-Text Dialogue:** Engage in interactive conversations about images, gaining insights and information.
*   **Advanced Vision-Language Alignment:**  Leverages a projection layer to align a frozen visual encoder (from BLIP-2) with the powerful Vicuna language model.
*   **Two-Stage Training:** Employs pre-training and fine-tuning for robust and reliable image understanding and generation.
*   **Efficient Fine-Tuning:** The second stage fine-tuning on a high-quality dataset improves usability.
*   **Open-Source & Accessible:**  Built upon open-source foundations for easy access and experimentation.

## Online Demo

Interact with MiniGPT-4 directly by uploading an image and chatting with the system.

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples and learn more on the [Project Page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News

*   **New Release:** A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! The demo requires as little as 12GB of GPU memory.

## Getting Started

### Installation

**1. Clone the Repository and Create Environment:**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare Vicuna Weights:**

*   **Download Pre-trained Weights (Optional):** Pre-trained weights may be available.
*   **Prepare Vicuna Weights (Manual):**
    *   MiniGPT-4 uses Vicuna-13B v0.1.1.
    *   Download the Vicuna delta weights from https://huggingface.co/lmsys/vicuna-13b-delta-v1.1:
        ```bash
        git lfs install
        git clone https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
        ```
    *   Obtain the original LLaMA-13B weights (instructions in the original README, including a way to download the weights.)
    *   Convert weights to Hugging Face Transformers format (detailed instructions in the original README).
    *   Use FastChat tools to create the final working weights (instructions in the original README).
    *   Set the path to the Vicuna weights in `minigpt4/configs/models/minigpt4.yaml` (line 16).

**3. Prepare Pre-trained MiniGPT-4 Checkpoint:**

*   Download the pre-trained checkpoint aligned with your chosen Vicuna model (13B or 7B - links provided in the original README).
*   Set the path to the checkpoint in `eval_configs/minigpt4_eval.yaml` (line 11).

### Run the Demo Locally

Execute the following command:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

*   Vicuna defaults to 8-bit loading for GPU memory optimization.
*   This configuration requires approximately 23GB (Vicuna 13B) or 11.5GB (Vicuna 7B) of GPU memory.

### Model Trimming (Optional)

*(Instructions for model trimming have been removed to keep the summary concise and the overall message clear)*

### Windows Deployment

For Windows deployment, please refer to this issue: https://github.com/Vision-CAIR/MiniGPT-4/issues/28

### Training

MiniGPT-4 is trained in two alignment stages:

**1. Stage 1: Pre-training**

*   Aligns the visual and language models using image-text pairs from Laion and CC datasets.
*   Download the dataset and prepare it according to the [Stage 1 Dataset Preparation](dataset/README_1_STAGE.md).
*   Run this command to begin training:
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```
*   The pre-trained MiniGPT-4 checkpoint is available [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2. Stage 2: Fine-tuning**

*   Fine-tunes MiniGPT-4 using a small, high-quality image-text dataset converted to a dialogue format.
*   Download and prepare the dataset according to [Stage 2 Dataset Preparation](dataset/README_2_STAGE.md).
*   Specify the Stage 1 checkpoint path in `train_configs/minigpt4_stage2_finetune.yaml`.
*   Run this command to start fine-tuning:
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Acknowledgements

*   **BLIP-2:** MiniGPT-4's model architecture follows BLIP-2.
*   **Lavis:** This repository is built upon Lavis.
*   **Vicuna:** The amazing language capabilities of Vicuna, a 13B parameter model.

## Citation

If you use MiniGPT-4 in your research, please cite it as follows:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## Community & Resources

*   **WeChat QR Code:** (For accessing the miniGPT-4 online version and various demo projects.)
*   **Knowledge Planet:**  (For learning about commercial AI.)
*   **License:** This repository uses the [BSD 3-Clause license](LICENSE.md). Code is largely based on [Lavis](https://github.com/salesforce/LAVIS) and the [BSD 3-Clause license](LICENSE_Lavis.md).

## Thanks
This project is forked from https://github.com/Vision-CAIR/MiniGPT-4
Most translations are based on https://github.com/Vision-CAIR/MiniGPT-4