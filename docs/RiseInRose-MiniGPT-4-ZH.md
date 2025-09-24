# MiniGPT-4: Unleashing Visual Language Understanding with Large Language Models

**MiniGPT-4 is a cutting-edge model that empowers you to converse with images, bridging the gap between vision and language.** Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology, MiniGPT-4 leverages advanced large language models to provide rich and informative image descriptions.

[View the original repository on GitHub](https://github.com/RiseInRose/MiniGPT-4-ZH)

## Key Features

*   **Image-to-Text Generation:**  Generate detailed and insightful textual descriptions from images.
*   **Interactive Dialogue:** Engage in conversations about images, exploring their content and context.
*   **Fine-tuned for High Quality:**  Achieves superior performance and usability through a two-stage training process.
*   **Efficient Training:** The second fine-tuning stage is computationally efficient, requiring only a single A100 for a short duration.
*   **Open-Source Foundation:** Built upon BLIP-2, Lavis, and the powerful Vicuna language model.
*   **Online Demo:** Experience MiniGPT-4 firsthand with our interactive online demo.

## Online Demo

Interact with MiniGPT-4 by uploading an image to learn more about its content!

[![Demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples and insights on the [project page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News

We now offer a pre-trained MiniGPT-4 aligned with Vicuna-7B!  GPU memory consumption for the demo can now be as low as 12GB.

## Core Concepts

MiniGPT-4 aligns a frozen visual encoder from BLIP-2 with the frozen LLM Vicuna via a projection layer. It is trained in two phases: 

1.  **Pre-training:**  The model undergoes a pre-training phase using approximately 5 million image-text pairs for around 10 hours on 4 A100 GPUs.
2.  **Fine-tuning:** A high-quality, smaller dataset created using the model and ChatGPT is used for fine-tuning on a conversational template, further enhancing generation quality and usability (single A100, 7 minutes).

## Getting Started

### Installation

**1. Code and Environment Setup**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare Vicuna Weights**

Follow the instructions [here](PrepareVicuna.md) to prepare the Vicuna weights, which can be obtained from [here](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1).  Alternatively, you can download the weights or use the one-click installation package.

**3. Prepare Pre-trained MiniGPT-4 Checkpoints**

Download pre-trained checkpoints for the respective Vicuna model version:

*   **Vicuna 13B:**  [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
*   **Vicuna 7B:**  [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

Set the path to the checkpoint in `eval_configs/minigpt4_eval.yaml` (line 11).

### Run the Demo Locally

Run the demo using:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

## Training

MiniGPT-4 training consists of two alignment stages, described above.
For training instructions, consult the original README.

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

## Community

Join the discussion and stay updated on MiniGPT-4:

*   **WeChat Group:**  Join via the QR code provided in the original README.
*   **Knowledge Planet:**  Access exclusive insights and experiences.
*   **Online Version** : Demo available and maintained by the author of the original repo.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md).  Many components are based on [Lavis](https://github.com/salesforce/LAVIS), with a BSD 3-Clause License [here](LICENSE_Lavis.md).