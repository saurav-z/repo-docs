# MiniGPT-4: Unlock Visual Understanding with Advanced Language Models

**MiniGPT-4 empowers you to understand images by connecting cutting-edge vision and language models.** Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology.  View the original repository [here](https://github.com/RiseInRose/MiniGPT-4-ZH).

## Key Features

*   **Image-to-Text Generation:**  Describe images in detail with human-like text.
*   **Question Answering:**  Get answers about the content of images.
*   **Interactive Dialogue:**  Engage in conversations about images, gaining deeper insights.
*   **Two-Stage Training:** Leverages a unique two-stage training process for optimal performance.
*   **Open-Source Foundation:** Built upon the robust BLIP-2 and Vicuna models.

##  Live Demo

Experience the power of MiniGPT-4 firsthand by chatting with the model about your images!

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples and functionalities on the [project page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

## Getting Started

### Installation

**1. Set up the Environment:**

Clone the repository, create and activate a Conda environment:

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare Vicuna Weights:**

*   Follow the instructions [here](PrepareVicuna.md) to download and prepare the Vicuna weights. You'll need to download the delta weights from lmsys and the original LLaMA weights, then apply the delta.  Consider using a torrent to download the LLaMA weights.

**3. Prepare MiniGPT-4 Checkpoint:**

Download the pre-trained checkpoints aligned with your chosen Vicuna model (13B or 7B):

|                                Checkpoint Aligned with Vicuna 13B                                |                               Checkpoint Aligned with Vicuna 7B                                |
|:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
 [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

**4. Run the Demo:**

Launch the local demo using:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

Adjust memory usage in `minigpt4_eval.yaml` for your GPU.

### Training

MiniGPT-4 training involves two stages:

**1. Stage 1: Pre-training**

Use image-text pairs from Laion and CC datasets to align the vision and language models. Prepare the dataset using instructions in `dataset/README_1_STAGE.md`. To start the first stage, run the following command.  This uses 4x A100 GPUs.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```
Download the Stage 1 checkpoint [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2. Stage 2: Fine-tuning**

Fine-tune MiniGPT-4 using a custom dataset converted to dialogue format.  Prepare your dataset based on instructions in `dataset/README_2_STAGE.md`.  Specify the Stage 1 checkpoint in  `train_configs/minigpt4_stage2_finetune.yaml`.  This uses 1x A100 GPU.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```
---

## Acknowledgements

*   **BLIP-2:** The model architecture is based on BLIP-2.
*   **LAVIS:**  The repository is built upon LAVIS.
*   **Vicuna:**  The powerful 13B parameter Vicuna model provides impressive language capabilities.

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

This repository is licensed under the [BSD 3-Clause License](LICENSE.md).
Code is based on [Lavis](https://github.com/salesforce/LAVIS), which has the [BSD 3-Clause License](LICENSE_Lavis.md).