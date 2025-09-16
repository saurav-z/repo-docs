# MiniGPT-4: Unleashing Visual Language Understanding with Large Language Models

**MiniGPT-4 brings vision and language together, enabling you to chat with images and unlock new insights.**  Find the original repository [here](https://github.com/RiseInRose/MiniGPT-4-ZH).

By Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology.

## Key Features

*   **Visual Language Understanding:** Processes images and generates descriptive text, answering questions, and providing insights.
*   **Advanced Architecture:** Leverages a frozen visual encoder (BLIP-2) and a frozen large language model (Vicuna) for efficient processing.
*   **Two-Stage Training:** Utilizes a pretraining phase with extensive image-text data, followed by a fine-tuning phase with high-quality, curated image-text pairs.
*   **Efficient Fine-tuning:** The second fine-tuning phase is computationally efficient and significantly improves the model's generation quality.
*   **Open Source & Accessible:** The project offers a demo, pre-trained models, and detailed instructions.

## Demo & Resources

*   **Online Demo:** Interact with MiniGPT-4 by uploading an image and initiating a conversation.
    [![demo](figs/online_demo.png)](https://minigpt-4.github.io)
*   **Project Page:** Explore more examples and learn more about MiniGPT-4.
    <a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
*   **Paper & Models:** Access the research paper and Hugging Face model.
    <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
*   **Colab Notebook & YouTube:** Run the code on Google Colab and watch the demo video.
    [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News

*   **Vicuna-7B Alignment:** A pre-trained MiniGPT-4 model aligned with Vicuna-7B is available, reducing demo GPU memory consumption to as low as 12GB.

## Getting Started

### Installation

1.  **Clone the Repository and Set Up Environment:**

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    conda env create -f environment.yml
    conda activate minigpt4
    ```

2.  **Prepare the Vicuna Weights:**

    *   Refer to [PrepareVicuna.md](PrepareVicuna.md) for instructions on preparing the Vicuna weights (v1.1).
    *   You'll need to download the delta weights from Hugging Face:  `git clone https://huggingface.co/lmsys/vicuna-13b-delta-v1.1`.
    *   You'll also need the original LLaMA-13B weights.  See the original README for details.
    *   **Alternative Download Sources**:  The original README contains alternative download methods, including IPFS and Baidu Netdisk.
    *   You'll need to install the fastchat package and convert Llama weights to huggingface format.

3.  **Prepare the MiniGPT-4 Checkpoint:**

    *   Download the pre-trained checkpoint aligned with either Vicuna 13B or 7B (links provided in the original README).
    *   Set the path to the checkpoint in the evaluation configuration file (`eval_configs/minigpt4_eval.yaml`).

### Local Demo

Run the demo locally with:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

## Training

MiniGPT-4 training consists of two alignment stages:

1.  **Stage 1: Pretraining**

    *   Train the model on image-text pairs from datasets like Laion and CC.  See [dataset/README_1_STAGE.md](dataset/README_1_STAGE.md) for dataset preparation.
    *   Run the pretraining script:

        ```bash
        torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
        ```

    *   Download the pretraining checkpoint [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

2.  **Stage 2: Fine-tuning**

    *   Fine-tune the model using a small, high-quality image-text dataset in a dialogue format.  See [dataset/README_2_STAGE.md](dataset/README_2_STAGE.md) for dataset preparation.
    *   Specify the Stage 1 checkpoint path in `train_configs/minigpt4_stage2_finetune.yaml`.
    *   Run the fine-tuning script:

        ```bash
        torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
        ```

## Acknowledgements

*   [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

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

This repository is licensed under the [BSD 3-Clause License](LICENSE.md). Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) and its [BSD 3-Clause License](LICENSE_Lavis.md).