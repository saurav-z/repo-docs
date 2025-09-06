# MiniGPT-4: Unleashing Visual Language Understanding with Large Language Models

**MiniGPT-4 empowers you to explore images by generating detailed descriptions and engaging in interactive conversations, bridging the gap between vision and language.**  ([Original Repo](https://github.com/RiseInRose/MiniGPT-4-ZH))

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny (King Abdullah University of Science and Technology)

**Key Features:**

*   **Advanced Visual Language Understanding:**  Leverages a frozen visual encoder from BLIP-2, aligning it with the powerful Vicuna LLM for enhanced image comprehension.
*   **Two-Stage Training:** Employs a two-stage training process, including pre-training on a large image-text dataset and fine-tuning on a curated high-quality dataset, to significantly improve generation quality and usability.
*   **Interactive Image Exploration:**  Engage in conversations with MiniGPT-4 about images, receiving detailed descriptions and answering your queries.
*   **Efficient Fine-tuning:** The second fine-tuning stage is computationally efficient, requiring only a single A100 GPU for a short duration.
*   **Open-Source & Accessible:** This project builds on top of open-source models and tools like BLIP-2, Lavis, and Vicuna, making it accessible for both research and practical applications.

## Resources

*   **Online Demo:** Interact with the model directly through a web-based demo.

    [![demo](figs/online_demo.png)](https://minigpt-4.github.io)

*   **Project Page:**  Find more examples and detailed information on the project page.
    <a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

*   **Paper:** Read the research paper for an in-depth understanding of the model.
    <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>

*   **Hugging Face Spaces:** Explore MiniGPT-4 through Hugging Face Spaces.
    <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

*   **Hugging Face Model:** Access the model on Hugging Face.
    <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

*   **Colab:**  Run the code in Google Colab.
    [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)

*   **YouTube:** Watch a demonstration video.
    [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

## News

*   A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available, with demo GPU memory consumption as low as 12GB.

---
## Installation

### Prerequisites

*   Python 3.7+
*   CUDA-enabled GPU (recommended)

### Steps:

1.  **Clone the repository and set up the environment:**

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    conda env create -f environment.yml
    conda activate minigpt4
    ```

2.  **Prepare the Vicuna weights:**

    *   Follow the instructions provided in [PrepareVicuna.md] to download and prepare the Vicuna weights.
    *   **Important:** Ensure you have the correct version of Vicuna (13B v1.1 or 7B) matching the provided checkpoint.

3.  **Prepare the MiniGPT-4 checkpoints:**

    *   Download the pre-trained checkpoint aligned with your chosen Vicuna model.
        *   Vicuna 13B:  [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
        *   Vicuna 7B:  [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)
    *   Set the checkpoint path in the evaluation configuration file `eval_configs/minigpt4_eval.yaml` (line 11).

### Local Demo

*   Run the demo on your local machine:

    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    ```

    *   The default configuration uses 8-bit loading for Vicuna to save GPU memory.  Adjust `low_resource` in `minigpt4_eval.yaml` if your GPU has more memory and set `low_resource: False`.

### Training

MiniGPT-4 training consists of two stages:

1.  **Stage 1: Pretraining** (Aligning visual and language models).  Follow the dataset preparation instructions in `dataset/README_1_STAGE.md`.

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

2.  **Stage 2: Fine-tuning** (Further aligning MiniGPT-4 through dialogue format).  Follow the dataset preparation instructions in `dataset/README_2_STAGE.md`.

    *   Specify the Stage 1 checkpoint path in `train_configs/minigpt4_stage2_finetune.yaml`.

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

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

## License

*   This project is licensed under the [BSD 3-Clause License](LICENSE.md).
*   Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) - [BSD 3-Clause License](LICENSE_Lavis.md).