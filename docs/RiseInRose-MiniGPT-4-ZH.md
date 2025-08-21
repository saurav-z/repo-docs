# MiniGPT-4: Supercharging Vision-Language Understanding with Large Language Models

**Unleash the power of visual understanding with MiniGPT-4, a groundbreaking model that bridges the gap between images and text!**  This project, by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology, leverages advanced large language models to analyze and describe images with remarkable accuracy.

[Original Repository](https://github.com/RiseInRose/MiniGPT-4-ZH)

## Key Features

*   **Image-to-Text Generation:** Generate detailed and informative text descriptions from images.
*   **Visual Question Answering:**  Answer questions about the content of images.
*   **Two-Stage Training:**  Employs a pre-training and fine-tuning process for optimal performance.
*   **Integration with BLIP-2 and Vicuna:**  Combines the strengths of a frozen visual encoder (BLIP-2) with the powerful Vicuna LLM.
*   **User-Friendly Demo:**  Interact with the model through an online demo to explore its capabilities.
*   **Open Source and Accessible:**  Provides resources and instructions for easy setup and experimentation.

## Quick Start

### Online Demo
Explore MiniGPT-4's capabilities by chatting with the model about your images:

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Find more examples on the [Project Page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

### Installation
Get started with MiniGPT-4 by following these steps:

**1. Set up the environment.**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```
**2. Prepare the Vicuna Weights**

Refer to the instructions in [PrepareVicuna.md](PrepareVicuna.md) to prepare Vicuna weights. Instructions are provided to download Llama and Vicuna delta weights.

**3. Prepare the MiniGPT-4 Checkpoints**
Download pre-trained checkpoints based on the Vicuna model used. Links are provided in the original README.

### Training

MiniGPT-4 undergoes a two-stage training process:

**1. Stage 1: Pre-training**
   - Use image-text pairs from the Laion and CC datasets.  See [dataset/README_1_STAGE.md](dataset/README_1_STAGE.md) for dataset preparation.
   - Run the training script using:
     ```bash
     torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
     ```

**2. Stage 2: Fine-tuning**
   - Use a custom, high-quality image-text dataset in a dialogue format. See [dataset/README_2_STAGE.md](dataset/README_2_STAGE.md).
   - Specify the Stage 1 checkpoint path in [train_configs/minigpt4_stage2_finetune.yaml](train_configs/minigpt4_stage2_finetune.yaml) and run:
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

This repository is licensed under the [BSD 3-Clause License](LICENSE.md).  Code is also based on [Lavis](https://github.com/salesforce/LAVIS), which is licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).

## Community and Support

*   Join the [Domestic AI application discussion group](#国内交流群) for support and updates.

## Thanks
Project forked from https://github.com/Vision-CAIR/MiniGPT-4
Most of the translation from https://github.com/Vision-CAIR/MiniGPT-4