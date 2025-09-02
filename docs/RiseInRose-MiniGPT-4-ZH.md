# MiniGPT-4: Unlock Advanced Vision-Language Understanding

**Experience the power of visual language with MiniGPT-4, a cutting-edge model that bridges the gap between images and text, enabling you to chat with images and extract detailed information.**  [Explore the original repository here.](https://github.com/RiseInRose/MiniGPT-4-ZH)

**(Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology.)***

## Key Features:

*   **Image-to-Text Dialogue:** Engage in interactive conversations about images.
*   **Advanced Vision-Language Alignment:** Leverages a projection layer to connect a frozen visual encoder (BLIP-2) with a frozen LLM (Vicuna).
*   **Two-Stage Training:**  Pre-trained on extensive image-text pairs and fine-tuned on a high-quality, curated dataset.
*   **High-Quality Output:** Generates coherent and informative text descriptions and answers about images.
*   **Open Source & Accessible:**  Leverages open-source models like Vicuna, and provides easy-to-use demo and Colab notebooks.

## Demo and Examples:

*   **Interactive Demo:** [Try the online demo here!](https://minigpt-4.github.io) (Click the image to chat with MiniGPT-4)
    [![demo](figs/online_demo.png)](https://minigpt-4.github.io)
*   **Project Page:** Find more examples and information on the [Project Page](https://minigpt-4.github.io).

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://minigpt-4.github.io)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](MiniGPT_4.pdf)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Vision-CAIR/MiniGPT-4)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News:

*   **Vicuna-7B Alignment:** A pre-trained MiniGPT-4 version aligned with Vicuna-7B is now available, reducing GPU memory requirements to as low as 12GB.

## Getting Started:

### Installation:

**1. Clone the Repository and Create Environment:**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare Vicuna Weights:**

*   **Download Pre-trained Weights:** (Due to potential copyright issues, weights might be available within the provided resources.) Or use your own resources.
*   **Prepare Vicuna Weights:** Detailed instructions are available in `PrepareVicuna.md` to prepare the Vicuna weights (based on Vicuna-13B v1.1).  This involves obtaining the delta weights from [lmsys/vicuna-13b-delta-v1.1](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1) and the original LLaMA-13B weights.

    *   **Alternative LLaMA Weights Source:** Instructions on how to acquire the necessary weights, including a torrent file link for LLaMA-13B.
*   **Convert LLaMA Weights:**  Use the provided conversion script from the Hugging Face Transformers library.

    ```bash
    # Install dependencies (using original pip source for faster installs):
    pip install transformers[sentencepiece]
    # Convert Weights:
    python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
    ```
*   **Create Working Weights:** Utilize the `fastchat.model.apply_delta` tool to generate the final working weights.
*   **Configure Model:** Set the path to your Vicuna weights within the model configuration file (e.g., `minigpt4/configs/models/minigpt4.yaml#L16`).

**3. Prepare MiniGPT-4 Checkpoint:**

*   **Download Checkpoint:** Download the pre-trained checkpoint aligned with your chosen Vicuna model (13B or 7B):

    *   [Checkpoint for Vicuna 13B](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
    *   [Checkpoint for Vicuna 7B](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)
*   **Configure Checkpoint Path:** Specify the path to the checkpoint file in the evaluation configuration file (e.g., `eval_configs/minigpt4_eval.yaml#L10`).

### Run the Demo Locally:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
```

### Model Trimming

*(Note: Model trimming can impact accuracy)*
*(Instructions on model trimming)*

## Training:

MiniGPT-4 training consists of two alignment stages.

**1. Stage 1: Pre-training**

*   Train the model on image-text pairs from the Laion and CC datasets.
*   Refer to `dataset/README_1_STAGE.md` for dataset preparation.
*   Run the training script:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

*   Download pre-trained checkpoint [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2. Stage 2: Fine-tuning**

*   Fine-tune the model on a curated image-text dataset in a dialogue format.
*   Refer to `dataset/README_2_STAGE.md` for dataset preparation.
*   Specify the Stage 1 checkpoint path in `train_configs/minigpt4_stage2_finetune.yaml`.
*   Run the fine-tuning script:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Acknowledgements:

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

## Citation:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## Community:

*(Information on joining a community and knowledge share)*

*   Follow us on [Wechat public account] to get the latest information.

## License:

This repository is licensed under the [BSD 3-Clause License](LICENSE.md). Code based on [Lavis](https://github.com/salesforce/LAVIS) is also covered by the [BSD 3-Clause License](LICENSE_Lavis.md).