# MiniGPT-4: Unleashing Advanced Vision-Language Understanding

**MiniGPT-4 empowers you to chat with images using cutting-edge AI, bridging the gap between visual and textual information.** This document provides an overview and guide for using MiniGPT-4, based on the original repository at [https://github.com/RiseInRose/MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH).

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny (King Abdullah University of Science and Technology).

## Key Features

*   **Image-Based Conversations:** Interact with MiniGPT-4 by uploading images and asking questions.
*   **Advanced Vision-Language Understanding:** Leveraging large language models (LLMs) for comprehensive image analysis and description.
*   **Two-Stage Training:**  Utilizes a pretraining and finetuning process for enhanced performance and usability.
*   **Open-Source and Accessible:**  Built upon the foundations of BLIP-2, Lavis, and Vicuna, with readily available resources.
*   **Hugging Face Integration:** Explore the model and related resources directly on Hugging Face Spaces and Models.

## Online Demo

Experience MiniGPT-4's capabilities firsthand!

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples on the [project page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## Getting Started

### 1. Installation

*   **Clone the Repository and Set Up Environment:**

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    conda env create -f environment.yml
    conda activate minigpt4
    ```

### 2. Prepare Pre-trained Models

*   **Vicuna Weights:** The current version of MiniGPT-4 is built upon Vicuna-13B v0.  Follow the instructions to prepare the Vicuna weights.
*   **MiniGPT-4 Checkpoints:** Download the pre-trained checkpoints aligned with your chosen Vicuna version (13B or 7B).

    *   **13B Checkpoint:** [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
    *   **7B Checkpoint:** [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

### 3. Run the Demo

*   **Local Demo:** Run the following command to test the demo locally:

    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    ```

## Training (Advanced)

MiniGPT-4 training involves two stages:

1.  **Stage 1: Pre-training:** Uses image-text pairs to align the visual and language models.  Follow the instructions to prepare and run the pre-training.
2.  **Stage 2: Finetuning:** Utilizes a smaller, high-quality dataset in a dialogue format to improve usability.

## Acknowledgements

*   **BLIP-2:**  MiniGPT-4's architecture is based on BLIP-2.
*   **Lavis:** The project is built upon the foundation of Lavis.
*   **Vicuna:**  Thanks to Vicuna for its impressive language capabilities.

## Citation

If you use MiniGPT-4 in your research, please cite the following:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This repository is licensed under the [BSD 3-Clause License](LICENSE.md).
The code is based on [Lavis](https://github.com/salesforce/LAVIS), licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).

## Community and Resources

*   Join the discussion for commercial AI applications.

    |              Follow for Updates               |                      Community Forum                       |
    |:-------------------------------:|:-----------------------------------------------:|
    | <img src="./img/qrcode.png" width="300"/> |  <img src="./img/WechatIMG81.jpeg" width="300"/> |

## Disclaimer
This project is a fork of https://github.com/Vision-CAIR/MiniGPT-4.