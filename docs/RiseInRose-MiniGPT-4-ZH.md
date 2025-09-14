# MiniGPT-4: Revolutionizing Vision-Language Understanding with Large Language Models

**MiniGPT-4 enhances visual language understanding by aligning a frozen visual encoder from BLIP-2 with the powerful Vicuna LLM, allowing you to chat about images!**

[View the original repository on GitHub](https://github.com/RiseInRose/MiniGPT-4-ZH)

## Key Features

*   **Image Chat:** Converse with MiniGPT-4 about any image, gaining insights and information.
*   **Two-Stage Training:** Leverages a two-stage training process, including a novel method for creating high-quality image-text pairs, to improve performance.
*   **High-Quality Generation:** Generates coherent and usable responses through a fine-tuned second stage.
*   **Open-Source & Accessible:** Built upon open-source models like BLIP-2 and Vicuna, making it accessible for research and experimentation.
*   **Efficient Fine-tuning:** The second stage fine-tuning is compute-efficient and can be done with a single A100 GPU in approximately 7 minutes.
*   **Local Demo:** Quickly test the model locally with a provided demo script.

## Online Demo

Interact with MiniGPT-4 directly through the online demo to understand images.

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples on the [Project Page](https://minigpt-4.github.io).

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://minigpt-4.github.io)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](MiniGPT_4.pdf)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Vision-CAIR/MiniGPT-4)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

## Getting Started

### Installation

**1. Clone the Repository and Set Up Environment**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare Pre-trained Vicuna Weights**

(Important: The original README contains significant detail about preparing Vicuna weights, including downloading the LLaMA weights, using git-lfs to download the delta weights, and running a script to apply the delta weights.)

*   Follow the instructions [here](PrepareVicuna.md) to obtain and prepare the Vicuna weights.  This involves obtaining the base LLaMA-13B weights, the Vicuna delta weights, and then applying the delta.  Alternatively, you can [download pre-trained Vicuna weights from here](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) or the 7B version [here](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing).

**3. Prepare Pre-trained MiniGPT-4 Checkpoints**

*   The pre-trained checkpoints can be downloaded from the links provided in the original README (see "2. 准备预训练的MiniGPT-4检查点" section).
*   Set the path to the pre-trained checkpoint in `eval_configs/minigpt4_eval.yaml` (line 11).

### Running the Local Demo

1.  Run the following command to start the local demo:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
```

(Remember to adjust the `--gpu-id` flag if needed)

### Model Trimming
The original README contained information on model trimming to save GPU memory and potentially improve performance.

(This section is summarized. Read the original README for detailed information).

### Training

MiniGPT-4 training includes two stages:

**1. First Stage Pre-training**

*   Trains the model using image-text pairs from Laion and CC datasets.
*   To prepare the datasets, see the [Stage 1 dataset preparation instructions](dataset/README_1_STAGE.md).
*   To start training, run the following command (example using 4 A100 GPUs):

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

**2. Second Stage Fine-tuning**

*   Uses a small, high-quality image-text dataset in a dialogue format for further alignment.
*   Prepare the dataset by following the [Stage 2 dataset preparation instructions](dataset/README_2_STAGE.md).
*   Specify the checkpoint path from the first stage in `train_configs/minigpt4_stage2_finetune.yaml`.
*   Run this command to start the finetuning:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Acknowledgements

*   [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [LAVIS](https://github.com/salesforce/LAVIS)
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

This project is licensed under the [BSD 3-Clause License](LICENSE.md).

The core code is based on [Lavis](https://github.com/salesforce/LAVIS), which is licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).