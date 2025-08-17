# MiniGPT-4: Unleash Visual Language Understanding with Advanced LLMs

**Effortlessly converse with images using MiniGPT-4, a groundbreaking model that combines the power of visual encoders with large language models, offering unparalleled image understanding capabilities.**  Learn more about this innovative approach with a link to the original repository: [https://github.com/RiseInRose/MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH)

*Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology.*

## Key Features

*   **Enhanced Visual Language Understanding:** MiniGPT-4 excels at interpreting and responding to visual content, providing detailed and insightful descriptions.
*   **Two-Stage Training:** The model employs a two-stage training process involving pretraining and fine-tuning to optimize performance.
*   **High-Quality Data for Fine-tuning:**  A curated dataset is used to enhance the reliability and usability of the model, improving conversational capabilities.
*   **Efficient Training:**  The fine-tuning stage is computationally efficient, requiring minimal resources.
*   **Emerging Capabilities:** MiniGPT-4 demonstrates capabilities similar to those found in GPT-4, making it a powerful tool for various applications.

## Online Demo
Interact with MiniGPT-4 directly!  Upload an image and start a conversation to explore its image understanding capabilities.
[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Find more examples on the [project page](https://minigpt-4.github.io).

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://minigpt-4.github.io)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](MiniGPT_4.pdf)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Vision-CAIR/MiniGPT-4)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News
A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! The demo's GPU memory consumption can be as low as 12GB.

## Getting Started

### Installation
1.  **Clone the repository and create a conda environment:**

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    conda env create -f environment.yml
    conda activate minigpt4
    ```

2.  **Prepare the Vicuna weights.**

    *Instructions for preparing Vicuna weights are included in the original README.*  (Original README instructions follow, or link to the original repo for the most up-to-date instructions.)

3.  **Prepare the pre-trained MiniGPT-4 checkpoint.**

    *The checkpoints can be downloaded as per the original README.* (Original README instructions follow, or link to the original repo for the most up-to-date instructions.)

### Run the Demo Locally
Run the following command to start the demo:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

*Configuration options for memory optimization are available in the original README.* (Original README instructions follow, or link to the original repo for the most up-to-date instructions.)

### Training
MiniGPT-4 training involves two alignment stages.

**1. Stage 1 Pretraining:**
*Instructions for preparing the dataset and running the first stage are detailed in the original README.* (Original README instructions follow, or link to the original repo for the most up-to-date instructions.)

**2. Stage 2 Fine-tuning:**
*Instructions for preparing the dataset, configuring the checkpoint path, and running the fine-tuning stage are detailed in the original README.* (Original README instructions follow, or link to the original repo for the most up-to-date instructions.)

## Acknowledgements
*   **BLIP2:** MiniGPT-4's architecture builds upon BLIP-2.
*   **Lavis:** This repository is built upon Lavis!
*   **Vicuna:** The amazing language capabilities of Vicuna (13B parameters) are truly impressive.

## Citation
```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## Community
*   Join the community for discussions and updates. (Original README instructions for joining the community follow)

## License
This repository is licensed under the [BSD 3-Clause License](LICENSE.md).

## Thanks
This project is forked from [https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
Most of the translation comes from [https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)