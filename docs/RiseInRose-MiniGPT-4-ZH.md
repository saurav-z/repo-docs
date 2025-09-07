# MiniGPT-4: Unleash the Power of Visual Language Understanding

**MiniGPT-4 seamlessly blends visual understanding with advanced language models to generate detailed image descriptions and engage in insightful conversations about images.** This README provides a comprehensive guide to understanding, using, and contributing to the MiniGPT-4 project.  Find the original repository [here](https://github.com/RiseInRose/MiniGPT-4-ZH).

## Key Features

*   **Enhanced Visual Language Understanding:** Leverages a frozen visual encoder from BLIP-2 and a frozen LLM (Vicuna) for superior image understanding.
*   **Two-Stage Training:**  Utilizes a pretraining phase with large image-text datasets and a finetuning phase with a curated high-quality dataset for improved performance.
*   **High-Quality Output:** Generates detailed and coherent image descriptions, exhibiting capabilities similar to GPT-4.
*   **Local and Online Demo:**  Easily test the model through both a local setup and an interactive online demo.
*   **Open Source & Community Focused:**  Provides training scripts, model checkpoints, and comprehensive documentation to foster community contributions.

## Online Demo

Interact with MiniGPT-4 and explore your images!

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

## Additional Resources

*   **Project Page:** [https://minigpt-4.github.io](https://minigpt-4.github.io)
*   **Paper (PDF):** [MiniGPT_4.pdf](MiniGPT_4.pdf)
*   **Hugging Face Spaces:**
    *   [Vision-CAIR/minigpt4 (Demo)](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
    *   [Vision-CAIR/MiniGPT-4 (Model)](https://huggingface.co/Vision-CAIR/MiniGPT-4)
*   **Colab Notebook:** [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
*   **YouTube Demo:** [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## Updates

*   **Pre-trained MiniGPT-4 with Vicuna-7B:** A new pre-trained model aligned with Vicuna-7B is now available, reducing GPU memory consumption to as low as 12GB.

## Getting Started

### 1. Installation

**1.1. Clone the Repository and Set up the Environment:**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**1.2. Prepare Vicuna Weights:**

Follow the instructions in [PrepareVicuna.md] (or find the relevant information extracted below) to prepare the Vicuna model weights.  Alternatively, download the pre-prepared weights (note that due to licensing, these weights are not directly distributed in this repo; see the original README for details, or search for them online).

### Preparing Vicuna Weights (Summary)

1.  **Download Delta Weights:** Download the Vicuna-13B delta weights from [lmsys/vicuna-13b-delta-v1.1](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1) using git-lfs.
2.  **Get LLaMA Weights:** Obtain the original LLaMA-13B weights (available via various sources, including the links and instructions in the original README, such as using torrents or IPFS).
3.  **Convert LLaMA Weights:** Convert the LLaMA weights to Hugging Face Transformers format using `convert_llama_weights_to_hf.py` (refer to the original README for specific steps).
4.  **Apply Delta:** Use the `fastchat.model.apply_delta` script to apply the delta weights to the base LLaMA weights.
5.  **Configure Paths:** Set the path to the Vicuna weights in `minigpt4/configs/models/minigpt4.yaml` (line 16).

**Example Commands (adapt paths as needed):**
```bash
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
# Get the LLaMA base weights - see links in original README
# Example conversion:
python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
# Apply delta:
python -m fastchat.model.apply_delta --base /path/to/llama-13b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-13b-delta-v0/
```

**Troubleshooting:**  The original README and community resources contain detailed troubleshooting steps for common errors during weight preparation (e.g., memory issues, tokenizer problems).  Consult the original README for the details.

**1.3. Prepare Pre-trained MiniGPT-4 Checkpoint:**

Download the pre-trained checkpoint aligned with your chosen Vicuna model (13B or 7B). Links are provided in the original README, or download from elsewhere (as indicated by the original README).
*   Set the path to the checkpoint in `eval_configs/minigpt4_eval.yaml` (line 11).

### 2. Running the Demo Locally

Run the following command to launch the local demo:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

**Note:**  The default configuration uses 8-bit loading for Vicuna to conserve GPU memory (approximately 23GB for Vicuna 13B and 11.5GB for Vicuna 7B). Adjust settings in `minigpt4_eval.yaml` (e.g., set `low_resource: False`) for more powerful GPUs to run in 16-bit mode.

### 3. Model Clipping (Optional)

The original README provides basic guidance, which is omitted here for conciseness, regarding methods to trim the model size (which may affect accuracy).  See the original README for these details.

## Training

MiniGPT-4 is trained in two stages:

**1. Stage 1: Pretraining**
*   Train on image-text pairs from Laion and CC datasets.
*   See `dataset/README_1_STAGE.md` for dataset preparation.
*   Run: `torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml`

**2. Stage 2: Finetuning**
*   Finetune on a smaller, high-quality image-text dataset in a conversational format.
*   See `dataset/README_2_STAGE.md` for dataset preparation.
*   Specify the Stage 1 checkpoint path in `train_configs/minigpt4_stage2_finetune.yaml`.
*   Run: `torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml`

## Acknowledgements

*   **BLIP-2:** (Model Architecture)
*   **Lavis:** (Framework)
*   **Vicuna:** (LLM - base model)

## Citation

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## Community and Support

*   **Join the AI Community (Links provided in original README):** Find a Chinese-speaking AI community group for support, discussions, and updates (links in the original README).

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md).  The code is based on [Lavis](https://github.com/salesforce/LAVIS), which uses the BSD 3-Clause License (LICENSE_Lavis.md).