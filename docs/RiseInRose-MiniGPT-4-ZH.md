# MiniGPT-4: Unleash Visual Language Understanding with Large Language Models

**MiniGPT-4 empowers you to have detailed conversations about images, blending vision and language seamlessly.**  Explore the cutting edge of AI with this innovative project!  ([Original Repo](https://github.com/RiseInRose/MiniGPT-4-ZH))

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny (King Abdullah University of Science and Technology)

## Key Features

*   **Visual Question Answering:**  Get detailed answers about what's in your images.
*   **Image Captioning:**  Generate descriptive captions for your visual content.
*   **Dialogue Generation:** Engage in interactive conversations about images, going beyond simple descriptions.
*   **Two-Stage Training:**  Leverages a two-stage training process for enhanced performance, involving pre-training and fine-tuning.
*   **Efficient Fine-tuning:** The second fine-tuning stage is computationally efficient, even on a single GPU.
*   **Open Source:** MiniGPT-4 is based on BLIP-2, Lavis, and Vicuna, all open-source projects.

## Online Demo

Interact with MiniGPT-4 directly! Click the image below to start a conversation about an image.

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples and detailed project information on the [Project Page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## What's New

*   A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! GPU memory consumption for the demo can be as low as 12GB.

## Getting Started

### Installation

**1. Prepare Code and Environment**

Clone the repository, create a Python environment, and activate it using the following commands:

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare Pre-trained Vicuna Weights**

*Instructions for preparing Vicuna weights are located [here](PrepareVicuna.md).*

**3. Prepare Pre-trained MiniGPT-4 Checkpoint**

Download pre-trained checkpoints based on your selected Vicuna model (13B or 7B):

*   **Checkpoint Aligned with Vicuna 13B:** [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
*   **Checkpoint Aligned with Vicuna 7B:** [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

### Running the Demo Locally

Run the demo on your local machine using:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
```

To save GPU memory, Vicuna loads in 8-bit by default with a search width of 1.

### Training

MiniGPT-4 utilizes a two-stage training process.

**1. Stage 1: Pre-training**

The model is pre-trained on image-text pairs from Laion and CC datasets.  See [dataset/README_1_STAGE.md](dataset/README_1_STAGE.md) for dataset preparation.

To start Stage 1 training:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

**2. Stage 2: Fine-tuning**

Fine-tuning is performed on a smaller, high-quality image-text dataset converted into a dialogue format. See [dataset/README_2_STAGE.md](dataset/README_2_STAGE.md) for details.  Specify the checkpoint path from Stage 1 within [train_configs/minigpt4_stage2_finetune.yaml](train_configs/minigpt4_stage2_finetune.yaml).

To start Stage 2 fine-tuning:

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

This project is licensed under the [BSD 3-Clause License](LICENSE.md). Code is based on [Lavis](https://github.com/salesforce/LAVIS) which is licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).