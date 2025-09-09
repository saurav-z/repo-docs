# MiniGPT-4: Unleash Visual Language Understanding with Large Language Models

**MiniGPT-4 leverages cutting-edge large language models to provide a powerful visual language understanding experience.**  ([Original Repo](https://github.com/RiseInRose/MiniGPT-4-ZH))

Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology.

## Key Features

*   **Enhanced Visual Understanding:**  MiniGPT-4 excels at understanding and describing the content of images.
*   **Two-Stage Training:**  A two-stage training process, involving pre-training and fine-tuning, optimizes performance.
*   **Efficient Fine-tuning:**  The second-stage fine-tuning is computationally efficient, requiring only a single A100 GPU for a short period.
*   **Open-Source Foundation:** Built upon the BLIP-2 architecture and leverages the powerful Vicuna language model.
*   **Interactive Demo:**  Experience MiniGPT-4's capabilities firsthand through an online demo.
*   **Model Alignment:**  Aligns a frozen visual encoder from BLIP-2 with the frozen LLM Vicuna through a projection layer.

## Demo & Resources

*   **Online Demo:**  [Interact with MiniGPT-4](https://minigpt-4.github.io)
    [![demo](figs/online_demo.png)](https://minigpt-4.github.io)
*   **Project Page:** [Explore further examples](https://minigpt-4.github.io)
    [![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://minigpt-4.github.io)
*   **Paper:**  [Read the full paper](MiniGPT_4.pdf)
    [![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](MiniGPT_4.pdf)
*   **Hugging Face Spaces:** [MiniGPT-4 on Hugging Face Spaces](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
    [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
*   **Hugging Face Model:** [MiniGPT-4 Model on Hugging Face](https://huggingface.co/Vision-CAIR/MiniGPT-4)
    [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Vision-CAIR/MiniGPT-4)
*   **Colab:**  [Run on Google Colab](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
    [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
*   **YouTube:**  [Watch the Demo Video](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)
    [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## Updates
*   Now offering a pre-trained MiniGPT-4 aligned with Vicuna-7B! GPU memory consumption can be as low as 12GB for the demo.

## Quickstart Installation

Follow these steps to get started with MiniGPT-4.  (Original instructions simplified and re-organized, focusing on the key steps and removing some of the more in-depth technical details which are better left to the original README and linked resources)

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    ```

2.  **Create and Activate Conda Environment:**

    ```bash
    conda env create -f environment.yml
    conda activate minigpt4
    ```

3.  **Prepare Vicuna Weights:** The original README includes a complex and detailed guide to preparing Vicuna weights. See the original README for detailed steps; however, a good starting point is:

    *   Download Vicuna delta weights from: [lmsys/vicuna-13b-delta-v1.1](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1).
    *   Download the original LLaMA-13B weights (instructions in the original README).
    *   Use the provided scripts to convert LLaMA weights to Hugging Face Transformers format.
    *   Apply the delta weights to create the final working Vicuna weights.

4.  **Prepare MiniGPT-4 Checkpoint:** (The original README provides links to download the pre-trained checkpoints. Choose the one aligned with your chosen Vicuna model, either 13B or 7B)

    *   13B Checkpoint: [Download from Drive](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
    *   7B Checkpoint: [Download from Drive](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

5.  **Run the Demo:**
    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    ```

## Training

MiniGPT-4's training involves two stages:

1.  **Stage 1: Pre-training:** Aligns visual and language models using image-text pairs from Laion and CC datasets. Use the original README's dataset preparation instructions to prepare the dataset.
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

    *   Download a pre-trained checkpoint [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link)
2.  **Stage 2: Fine-tuning:** Uses a smaller, high-quality image-text dataset (prepare as described in the original README).
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Acknowledgements

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2): Model architecture based on BLIP-2.
*   [LAVIS](https://github.com/salesforce/LAVIS):  The repository is built upon LAVIS.
*   [Vicuna](https://github.com/lm-sys/FastChat): The fantastic language capabilities of the open-source Vicuna model.

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
Code is also based on [Lavis](https://github.com/salesforce/LAVIS), which is also under the [BSD 3-Clause License](LICENSE_Lavis.md).