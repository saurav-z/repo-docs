# MiniGPT-4: Unleash Visual Language Understanding with Advanced LLMs

**Explore the world of visual language with MiniGPT-4, a powerful model that enhances image understanding using cutting-edge large language models. ** Explore the original repo: [https://github.com/RiseInRose/MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH)

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny (King Abdullah University of Science and Technology)

## Key Features

*   **Enhanced Visual Language Understanding:** MiniGPT-4 leverages advanced large language models to interpret and generate text descriptions based on images.
*   **Two-Stage Training:**  The model undergoes a two-stage training process, including a pre-training phase and a fine-tuning phase for improved accuracy and usability.
*   **High-Quality Dataset:** The model utilizes a specially curated dataset created using the model itself, enhancing the quality of generated image descriptions.
*   **Efficient Fine-tuning:** The second stage of fine-tuning is computationally efficient, requiring only a single A100 GPU for around 7 minutes.
*   **Open-Source and Accessible:** The project offers a variety of resources including online demos, Hugging Face integrations, and Colab notebooks for ease of use.

## Online Demo

Interact with the MiniGPT-4 model and gain insights into your images through our interactive online demo.

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples on the [project page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

## News

*   Pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! The demo now consumes as little as 12GB of GPU memory.

---

## Getting Started

### Installation

**1. Code and Environment Setup**

1.  Clone the repository and create a Python environment:

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    conda env create -f environment.yml
    conda activate minigpt4
    ```

**2. Prepare Vicuna Weights**

Follow the instructions [here](PrepareVicuna.md) to prepare the Vicuna weights.  (Directly download from [https://github.com/RiseInRose/MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH) for the Chinese version)

**3. Prepare the Pre-trained MiniGPT-4 Checkpoint**

Download the pre-trained checkpoint based on your prepared Vicuna model.

*   Checkpoint Aligned with Vicuna 13B: [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
*   Checkpoint Aligned with Vicuna 7B: [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

Then, in the evaluation configuration file [minigpt4\_eval.yaml](eval\_configs/minigpt4\_eval.yaml#L10), set the path for the pre-trained checkpoint on line 11.

### Launch Demo Locally

Run the following command to test the demo locally:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
```

The configuration requires approximately 23GB of GPU memory for Vicuna 13B and 11.5GB of GPU memory for Vicuna 7B.

### Training

MiniGPT-4's training consists of two alignment stages.

**1. First Stage Pre-training**

In the first pre-training stage, the model is trained with image-text pairs from the Laion and CC datasets to align the visual and language models. See the [first stage dataset preparation instructions](dataset/README_1_STAGE.md).

To start the first stage training, run the following command.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

**2. Second Stage Fine-tuning**

In the second stage, a high-quality image-text dataset in a dialogue format is utilized to fine-tune MiniGPT-4.

*   Specify the path to the checkpoint file from the 1st stage of training.
*   Run the following command.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Acknowledgements

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

## License

This repository is licensed under the [BSD 3-Clause License](LICENSE.md).

## Contact

*   For more information, or to join the community, see:
    *   [Wechat](https://github.com/RiseInRose/MiniGPT-4-ZH)
    *   [Knowledge Planet](https://github.com/RiseInRose/MiniGPT-4-ZH)

## Credits
Project Forked from: [https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)