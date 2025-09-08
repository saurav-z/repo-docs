# MiniGPT-4: Unleashing Visual Language Understanding with Large Language Models

**MiniGPT-4 leverages cutting-edge large language models to create an advanced visual language understanding system.**  Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology (KAUST). ([Original Repo](https://github.com/RiseInRose/MiniGPT-4-ZH))

## Key Features

*   **Enhanced Visual Language Understanding:**  MiniGPT-4 excels at interpreting and generating text based on visual input.
*   **Two-Stage Training:** Employs a pre-training and fine-tuning approach for optimal performance and alignment.
*   **Efficient Fine-Tuning:**  Achieves significant improvements with a small, high-quality dataset in the second training phase.
*   **Open Source:** Based on BLIP-2, Lavis, and Vicuna, all of which are open source projects.
*   **Online Demo:** Interact with MiniGPT-4 and explore its capabilities with an easy-to-use online demo.

## Online Demo

Chat with MiniGPT-4 by uploading an image and explore its capabilities.

[![Demo](figs/online_demo.png)](https://minigpt-4.github.io)

Find more examples on the [Project Page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News

*   A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! The demo's GPU memory consumption can be as low as 12GB.

## Getting Started

### Installation

**1.  Prepare the Code and Environment**

Clone the repository, create a Python environment, and activate it:

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2.  Prepare the Pre-trained Vicuna Weights**

*   Download pre-trained weights [Here](https://github.com/Vision-CAIR/MiniGPT-4#2-prepare-the-pre-trained-vicuna-weights).

    *   The current version of MiniGPT-4 is built on Vicuna-13B v0. Refer to the instructions [Here](PrepareVicuna.md) for preparing the Vicuna weights.

**3.  Prepare the Pre-trained MiniGPT-4 Checkpoint**

Download a pre-trained checkpoint based on your prepared Vicuna model:

|                                Checkpoint Aligned with Vicuna 13B                                |                               Checkpoint Aligned with Vicuna 7B                                |
| :------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
|  [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) |

### Run Demo Locally

Run the demo on your local machine:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

## Training

MiniGPT-4 training consists of two alignment stages.

**1. Stage 1 Pre-training**

*   Use image-text pairs from the Laion and CC datasets to train the model to align visual and language models.
*   Download and prepare the dataset by following the [Stage 1 Dataset Preparation Instructions](dataset/README_1_STAGE.md).
*   Start the first stage of training by running the following command.
```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```
*   The MiniGPT-4 checkpoint from the first stage of training is available for download [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2. Stage 2 Fine-tuning**

*   Use your own small, high-quality image-text pair dataset and convert it to a dialogue format to further align MiniGPT-4.
*   Prepare the dataset by following the [Stage 2 Dataset Preparation Instructions](dataset/README_2_STAGE.md).
*   Specify the path to the checkpoint file from the first stage of training in the [train_configs/minigpt4_stage2_finetune.yaml](train_configs/minigpt4_stage2_finetune.yaml) file.
*   Run the following command to start the second stage.
```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Acknowledgements

*   [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

## Citation

If you use MiniGPT-4 in your research, please cite it as:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This repository is licensed under the [BSD 3-Clause License](LICENSE.md).

## Contributions

Project fork from: https://github.com/Vision-CAIR/MiniGPT-4.
Most of the translation from: https://github.com/Vision-CAIR/MiniGPT-4