# MiniGPT-4: Revolutionizing Vision-Language Understanding with Advanced LLMs

**Effortlessly understand and interact with images using MiniGPT-4, a powerful vision-language model built to bridge the gap between visual and textual information.** [View the original repository here](https://github.com/RiseInRose/MiniGPT-4-ZH).

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny (King Abdullah University of Science and Technology)

## Key Features

*   **Visual-Language Alignment:** MiniGPT-4 utilizes a projection layer to align a frozen visual encoder (BLIP-2) with a frozen large language model (LLM) (Vicuna).
*   **Two-Stage Training:** A two-stage training process, including pre-training and fine-tuning on a high-quality dataset, enhances the model's ability to generate coherent and relevant descriptions.
*   **Enhanced Generative Capabilities:** Achieves emergent vision-language capabilities, enabling diverse image understanding and generation tasks.
*   **Reduced GPU Memory Consumption:** Optimized to run efficiently, with the latest Vicuna-7B aligned model capable of running on as little as 12GB of GPU memory.

## Online Demo

Experience the power of MiniGPT-4 firsthand. Simply upload an image and engage in a conversation to explore its contents.

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples on the [Project Page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News

*   **Vicuna-Aligned MiniGPT-4:** Now offering a pre-trained MiniGPT-4 aligned with Vicuna-7B, requiring as little as 12GB of GPU memory for the demo.

## Installation

### Prerequisites

*   **Clone the Repository:**

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    ```
*   **Create and Activate a Python Environment:**

    ```bash
    conda env create -f environment.yml
    conda activate minigpt4
    ```

### 1. Prepare Vicuna Weights

**Download Weights (Option 1 - Recommended):**

Download pre-prepared weights (may be available in WeChat group).

**Prepare Vicuna Weights (Option 2 - Manual):**

1.  **Download Vicuna Delta Weights:**
    ```bash
    git lfs install
    git clone https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
    ```
2.  **Obtain LLaMA-13B Weights:**  Acquire the original LLaMA-13B weights (details on how to get these are in the original README).  Consider using the provided links.
3.  **Convert Weights to Hugging Face Format:**
    ```bash
    cd FastChat
    git checkout v0.2.3
    pip install -e .
    pip install transformers[sentencepiece]
    ```
    Run conversion script (adapt paths as needed):
    ```bash
    python src/transformers/models/llama/convert_llama_weights_to_hf.py \
        --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
    ```
    *   **Troubleshooting:** Resolve potential errors during conversion.
4.  **Load Model and Tokenizer:**
    ```python
    from transformers import LlamaForCausalLM, LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("/output/path")
    model = LlamaForCausalLM.from_pretrained("/output/path")
    ```

5.  **Create Working Weights:**
    ```bash
    pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
    python -m fastchat.model.apply_delta --base /path/to/llama-13b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-13b-delta-v0/
    ```
    *   **Troubleshooting:** Address potential errors (e.g., "Tokenizer class LLaMATokenizer does not exist," "RuntimeError: The size of tensor a (32000) must match the size of tensor b (32001)").

6.  **Weight Structure:**  Organize the final weight files:
    ```
    vicuna_weights
    ├── config.json
    ├── generation_config.json
    ├── pytorch_model.bin.index.json
    ├── pytorch_model-00001-of-00003.bin
    ...
    ```
7.  **Set Vicuna Weight Path:**  Configure the path to your `vicuna_weights` folder in `minigpt4/configs/models/minigpt4.yaml` (line 16).

### 2. Prepare MiniGPT-4 Checkpoints

**Download Checkpoint:**  Download the pre-trained MiniGPT-4 checkpoint aligned with your chosen Vicuna model (13B or 7B). Links are provided in the original README.
**Set Checkpoint Path:**  Specify the path to the checkpoint in `eval_configs/minigpt4_eval.yaml` (line 11).

### 3. Run the Demo Locally

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

*   **GPU Memory Optimization:** Vicuna defaults to 8-bit loading with a search width of 1.

### Model Optimization

*   **Model Pruning (Optional):**
    *   [Instructions provided, but needs to be added.]
    *   Remember that model pruning may affect model accuracy.

### 4. Training

MiniGPT-4 is trained in two stages.

**1. Stage 1: Pre-training**

*   **Objective:** Align vision and language models.
*   **Dataset:** Image-text pairs from Laion and CC datasets (see [First-stage dataset preparation](dataset/README_1_STAGE.md) for details).
*   **Training Command:**
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```
*   **Checkpoint:**  A pre-trained checkpoint from Stage 1 is available for download.

**2. Stage 2: Fine-tuning**

*   **Objective:** Further align MiniGPT-4 using a high-quality dataset in a dialogue format.
*   **Dataset:**  A custom dataset of image-text pairs (see [Second-stage dataset preparation](dataset/README_2_STAGE.md) for details).
*   **Training Command:**
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```
*   **Configuration:**  Specify the path to the Stage 1 checkpoint and the output path in `train_configs/minigpt4_stage2_finetune.yaml`.

## Acknowledgements

*   [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [LAVIS](https://github.com/salesforce/LAVIS)
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

*   [Join WeChat Group (QR code)] - to get the latest updates, and share commercial applications of AI.
*   [Join the Zhishi Xingqiu] - Provides insights into AI for businesses and valuable resources for AI.

## License
This project is licensed under the [BSD 3-Clause License](LICENSE.md). Code is based on [Lavis](https://github.com/salesforce/LAVIS) and is licensed under the [BSD 3-Clause License](LICENSE_Lavis.md)