# MiniGPT-4: Unleash Visual Language Understanding with Advanced LLMs

**Experience a new era of AI: MiniGPT-4 transforms images into insightful text, powered by cutting-edge large language models.**  [Explore the original repository](https://github.com/RiseInRose/MiniGPT-4-ZH)

**Developed by:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny (King Abdullah University of Science and Technology)

## Key Features:

*   **Advanced Vision-Language Understanding:**  MiniGPT-4 leverages the power of large language models to analyze and describe images in detail.
*   **Two-Stage Training:** A pre-training phase using a large dataset followed by fine-tuning with a smaller, high-quality dataset, resulting in enhanced generation capabilities.
*   **High-Quality Image-Text Generation:** Generates coherent and usable text descriptions, demonstrating advanced visual language skills.
*   **Efficient Fine-tuning:**  The second fine-tuning stage is computationally efficient, requiring only a single A100 GPU for approximately 7 minutes.
*   **Open Source and Accessible:**  Leverages open-source models like Vicuna and BLIP-2, and the code is available on GitHub, making it easy for anyone to get started.

## Online Demo

[Click to Chat with MiniGPT-4](https://minigpt-4.github.io)

  [![demo](figs/online_demo.png)](https://minigpt-4.github.io)

  *Explore more examples on the [project page](https://minigpt-4.github.io).*

  [![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://minigpt-4.github.io)
  [![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](MiniGPT_4.pdf)
  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
  [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Vision-CAIR/MiniGPT-4)
  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
  [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News

*   **Pre-trained MiniGPT-4 with Vicuna-7B Alignment:** Experience reduced GPU memory consumption, now down to as low as 12GB.

## Getting Started

### Installation

**1. Clone the Repository and Set Up Environment:**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare Vicuna Weights:**

*   Instructions on preparing Vicuna weights can be found [here](PrepareVicuna.md). This involves downloading the delta weights and the original LLAMA-13B weights (details in the original README, including alternative download methods).
*   Note:  The provided instructions cover the necessary steps, including the conversion of weights using the Hugging Face Transformers library.
*   Prepare the weights according to the steps provided (including possible workarounds if you have hardware limitations)

**3. Prepare the Pre-trained MiniGPT-4 Checkpoint:**

*   Download the pre-trained checkpoint corresponding to your Vicuna model version (13B or 7B) from the provided links.
*   Specify the path to the checkpoint in the evaluation configuration file (`eval_configs/minigpt4_eval.yaml`).

### Local Demo

*   Run the demo locally using:

    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    ```

*   The default settings use 8-bit loading and a search width of 1 for memory efficiency (approximately 23GB of GPU memory for Vicuna 13B and 11.5GB for Vicuna 7B).
*   For higher-end GPUs, adjust the configuration file (`minigpt4_eval.yaml`) to disable `low_resource` and increase the search width for 16-bit operation.

### Training

MiniGPT-4 training is divided into two phases:

**1. Stage 1: Pre-training**

*   Trains the model using image-text pairs from Laion and CC datasets to align visual and language models.
*   Instructions for preparing the dataset are in `dataset/README_1_STAGE.md`.
*   Start the pre-training:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

*   A Stage 1 checkpoint is available [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2. Stage 2: Fine-tuning**

*   Fine-tunes MiniGPT-4 on a smaller, high-quality, image-text dataset in a dialog format.
*   Dataset preparation instructions are in `dataset/README_2_STAGE.md`.
*   Specify the Stage 1 checkpoint path in `train_configs/minigpt4_stage2_finetune.yaml`.
*   Start the fine-tuning:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Acknowledgements

*   [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2):  MiniGPT-4's architecture is based on BLIP-2.
*   [Lavis](https://github.com/salesforce/LAVIS):  This repository is built upon Lavis.
*   [Vicuna](https://github.com/lm-sys/FastChat):  Thanks to Vicuna for its impressive language capabilities.

## Citation

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

*   This repository is licensed under the [BSD 3-Clause License](LICENSE.md).
*   The code is partially based on [Lavis](https://github.com/salesforce/LAVIS), which is licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).