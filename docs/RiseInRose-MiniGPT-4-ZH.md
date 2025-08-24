# MiniGPT-4: Revolutionizing Visual Language Understanding with Large Language Models

**Unlocking the power of visual language, MiniGPT-4 allows you to chat with images and gain insights like never before.**  Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology, MiniGPT-4 leverages cutting-edge large language models to provide unparalleled visual comprehension.  ([Original Repo](https://github.com/RiseInRose/MiniGPT-4-ZH))

## Key Features

*   **Image-Based Chat:** Engage in conversations with images to understand their content and context.
*   **Enhanced Visual-Language Alignment:**  Leverages advanced techniques to align visual encoders with powerful LLMs like Vicuna.
*   **Two-Stage Training:**  Employs a pretraining and fine-tuning approach for robust performance and enhanced usability.
*   **High-Quality Dataset:** Utilizes a curated dataset for fine-tuning, ensuring reliable and coherent responses.
*   **Open-Source and Accessible:** Built upon open-source projects like BLIP-2, Lavis, and Vicuna.
*   **Model Versatility:** Comes with both 7B and 13B model weights for use across different hardware configurations.

## Quick Start

### 1. Online Demo

Try out the interactive demo and learn about the image with MiniGPT-4 by clicking the image below.

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Find more examples in the [Project Page](https://minigpt-4.github.io)

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

### 2. Installation

**1.  Set up the Code and Environment**

Clone the repository, create a Python environment, and activate it using:

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare the Vicuna Weights**

Download the pre-trained Vicuna weights, and the weights from the [link](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1) and follow the instructions, and also you can download LLaMA model weights, using the torrent from the original repo, or obtain it as described in the repo or by referring to the links:

*   [How to Prepare Vicuna Weights](PrepareVicuna.md)
*   [LLaMA weights using torrent](CDEE3052D85C697B84F4C1192F43A2276C0DAEA0.torrent)

**3. Prepare the MiniGPT-4 Checkpoint**

*   Download the checkpoint based on your Vicuna model:

    *   [MiniGPT-4 Checkpoint (aligned with Vicuna 13B)](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
    *   [MiniGPT-4 Checkpoint (aligned with Vicuna 7B)](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

*   Set the path of the pre-trained checkpoint in `eval_configs/minigpt4_eval.yaml` (line 11).

### 3. Run the Demo Locally

Run the demo on your local machine by using the following command:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

### 4. Training

MiniGPT-4 training includes the two alignment stages.

**1. Stage 1 Pretraining**

Train the model with image-text pairs.

To download and prepare the dataset, please check out the dataset README file [dataset/README_1_STAGE.md].
To start the stage 1 training, run the following command.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

**2. Stage 2 Fine-tuning**

Fine-tune the model with high-quality image-text pairs with a dialog format.
To download and prepare our stage 2 dataset, check out the dataset README file [dataset/README_2_STAGE.md].

To start the stage 2 training, first specify the checkpoint file path of stage 1 training in [train_configs/minigpt4_stage2_finetune.yaml].
Run the following command.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Acknowledgements

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

## Citation

If you use MiniGPT-4 in your research, please cite our work:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md).