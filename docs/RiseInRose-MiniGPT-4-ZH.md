# MiniGPT-4: Unleashing Visual Language Understanding with Large Language Models

**MiniGPT-4 empowers you to have insightful conversations about images using advanced large language models.** (Original repo: [https://github.com/RiseInRose/MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH))

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny.

**Affiliation:** King Abdullah University of Science and Technology (KAUST)

## Key Features

*   **Enhanced Visual Language Understanding:**  MiniGPT-4 leverages a projection layer to align a frozen visual encoder from BLIP-2 with the frozen LLM Vicuna.
*   **Two-Stage Training:** MiniGPT-4 is trained in two stages: a pre-training phase and a fine-tuning phase for superior performance.
*   **High-Quality Data:** The fine-tuning phase utilizes a curated dataset of high-quality image-text pairs, improving the model's generation reliability.
*   **Open-Source & Accessible:** Enjoy a user-friendly model that can generate detailed and coherent image descriptions.

## Demo

Explore MiniGPT-4's capabilities interactively!

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Find more examples and information on the [Project Page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News

*   A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! The demo's GPU memory consumption can be as low as 12GB.

## Getting Started

### Installation

**1. Clone the repository and set up the environment:**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare Vicuna Weights:**

*   Download the weights (details in the original README, or from the original repo).
*   Prepare Vicuna weights following the instructions: [PrepareVicuna.md](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/PrepareVicuna.md)

**3. Prepare MiniGPT-4 Checkpoints:**

*   Download the pre-trained checkpoints aligned with your chosen Vicuna model (13B or 7B). (Links provided in original README.)
*   Set the checkpoint path in the evaluation configuration file (`eval_configs/minigpt4_eval.yaml`).

### Run the Demo Locally

Launch the demo on your local machine:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

## Training

MiniGPT-4 undergoes two alignment stages:

**1. Stage 1: Pre-training**

*   Train the model using image-text pairs from the Laion and CC datasets.  (Dataset preparation details are in `dataset/README_1_STAGE.md`.)
*   Run the pre-training with:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

**2. Stage 2: Fine-tuning**

*   Fine-tune the model with a curated dataset of image-text pairs formatted in a conversational style. (Dataset preparation details are in `dataset/README_2_STAGE.md`.)
*   Specify the checkpoint path from Stage 1 in  `train_configs/minigpt4_stage2_finetune.yaml`.
*   Run the fine-tuning with:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Acknowledgements

*   [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

## Citation

If you use MiniGPT-4 in your research, please cite:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md). Many codes are based on [Lavis](https://github.com/salesforce/LAVIS), with a [BSD 3-Clause License](LICENSE_Lavis.md)

##  Additional Resources & Community
*  Explore online versions and find helpful resources, like AI-related updates, in the original README (links available).

##  Credits
Forked from [https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)