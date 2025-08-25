# MiniGPT-4: Unleash Visual Language Understanding with Large Language Models

**Explore the world through the eyes of AI!** MiniGPT-4 enhances visual language understanding by aligning a frozen visual encoder from BLIP-2 with the powerful Vicuna large language model.  [Visit the original repo](https://github.com/RiseInRose/MiniGPT-4-ZH).

**Key Features:**

*   **Visual-Language Alignment:** Leverages a frozen visual encoder (BLIP-2) and the Vicuna LLM for robust image understanding.
*   **Two-Stage Training:** Utilizes a two-stage training process, including a pre-training phase with 5 million image-text pairs and a fine-tuning phase on a high-quality, curated dataset, to maximize performance.
*   **Improved Generation and Usability:** The fine-tuning phase, using a novel approach with high-quality image-text pairs, dramatically improves generation reliability and overall user experience.
*   **Open-Source & Accessible:**  Built upon open-source foundations (BLIP-2, LAVIS, Vicuna) and includes detailed instructions for setup.
*   **Supports 7B and 13B Models:**  Provides checkpoints aligned with both the Vicuna-7B and Vicuna-13B models.

**Online Demo:**

Interact with MiniGPT-4 and explore the information contained within your images.

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Find more examples and explore the project at the [project page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

**News:**

*   Now offering a pre-trained MiniGPT-4 aligned with Vicuna-7B! GPU memory consumption can now be as low as 12GB for the demo.

---

**Getting Started:**

### Installation

**1. Code and Environment Setup:**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare the Pre-trained Vicuna Weights:**

Detailed instructions on preparing Vicuna weights are available [here](PrepareVicuna.md).  Alternatively, you can download the pre-prepared weights (details are available in the original README).

**3. Prepare the Pre-trained MiniGPT-4 Checkpoint:**

Download the checkpoint for your chosen Vicuna model (7B or 13B):

|                                Checkpoint Aligned with Vicuna 13B                                |                               Checkpoint Aligned with Vicuna 7B                                |
|:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
 [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

Set the path to your checkpoint in `eval_configs/minigpt4_eval.yaml` (line 11).

### Run the Demo Locally:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

---

**Training:**

MiniGPT-4 training involves two alignment stages:

*   **Stage 1: Pre-training:** Train the model on image-text pairs (Laion and CC datasets).  See [first-stage dataset preparation](dataset/README_1_STAGE.md) for more information. Start pre-training with:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

*   **Stage 2: Fine-tuning:** Fine-tune MiniGPT-4 using a curated dataset in a conversational format.  See [second-stage dataset preparation](dataset/README_2_STAGE.md). Specify the checkpoint path from Stage 1 and output path in `train_configs/minigpt4_stage2_finetune.yaml`. Start fine-tuning with:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

---

**Acknowledgements:**

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

---

**Citation:**

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

---

**License:**

This repository is licensed under the [BSD 3-Clause License](LICENSE.md). The code is based on [Lavis](https://github.com/salesforce/LAVIS), which is also licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).

---

**Contact and Support:**

*   (Optional) [Join the discussion group] (Links to the discussion group).