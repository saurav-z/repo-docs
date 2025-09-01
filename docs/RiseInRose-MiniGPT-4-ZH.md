# MiniGPT-4: Unleashing Visual Language Understanding with Advanced LLMs

**Effortlessly converse with images and extract insightful information using MiniGPT-4, a cutting-edge model that bridges the gap between vision and language.**  [Original Repository](https://github.com/RiseInRose/MiniGPT-4-ZH)

Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology.

## Key Features

*   **Advanced Visual Language Understanding:** MiniGPT-4 leverages a frozen visual encoder from BLIP-2 and a frozen LLM (Vicuna) to provide a deep understanding of images and their content.
*   **Two-Stage Training for Optimal Performance:** The model is trained in two stages: a pre-training phase using a large dataset of image-text pairs, followed by a fine-tuning phase on a smaller, high-quality dataset created with the help of ChatGPT.
*   **Efficient Fine-tuning:** The second fine-tuning stage is computationally efficient, requiring only a single A100 GPU and a short amount of time to significantly enhance generation capabilities and overall usability.
*   **Mimics GPT-4 capabilities:** Produces high-quality and insightful responses, similar to the advanced visual language understanding demonstrated by GPT-4.
*   **Easy to get started:** Instructions and a working environment can be set up in your own environment, or you can utilize community resources such as Colab and one-click install packages.

## Online Demo

Interact with MiniGPT-4 directly through its online demo!

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

## Getting Started

### 1. Installation

**Prerequisites:**
*   Clone the repository:
    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    ```
*   Create and activate a Python environment:
    ```bash
    conda env create -f environment.yml
    conda activate minigpt4
    ```

### 2. Prepare Vicuna Weights

  *  See instructions in [PrepareVicuna.md](PrepareVicuna.md) or use community-provided resources.

### 3. Prepare MiniGPT-4 Checkpoints

*   Download the pre-trained checkpoints:

    |                                Checkpoint Aligned with Vicuna 13B                                |                               Checkpoint Aligned with Vicuna 7B                                |
    :------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
     [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

### 4. Running the Demo Locally

*   Run the demo.  Adjust GPU use as required.

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

## Training

MiniGPT-4 training involves two alignment stages:

**1. Stage 1: Pre-training**

Train the model on image-text pairs to align visual and language models.

*   Prepare your dataset by following the instructions in [dataset/README_1_STAGE.md].
*   Run the pre-training:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

**2. Stage 2: Fine-tuning**

Fine-tune the model on a curated dataset for enhanced conversational ability.

*   Prepare your dataset following the instructions in [dataset/README_2_STAGE.md].
*   Configure the checkpoint path from Stage 1 in [train_configs/minigpt4_stage2_finetune.yaml].
*   Run the fine-tuning:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Acknowledgements

*   **BLIP-2:** The model architecture is inspired by BLIP-2.
*   **Lavis:** This repository is built upon Lavis.
*   **Vicuna:** The amazing language capabilities of Vicuna are used.

## Citation

If you use MiniGPT-4 in your research or applications, please cite the following:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This repository is licensed under the [BSD 3-Clause License](LICENSE.md). Many parts of the code are based on [Lavis](https://github.com/salesforce/LAVIS), which is under the [BSD 3-Clause License](LICENSE_Lavis.md).