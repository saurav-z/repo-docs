# MiniGPT-4: Unleash the Power of Visual Language Understanding

**Experience the future of AI: MiniGPT-4 seamlessly blends visual and language understanding, enabling you to chat with images and extract insightful information.**

**Original Repo:** [https://github.com/RiseInRose/MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH)

## Key Features

*   **Interactive Image Chat:** Engage in conversations with images to learn about their content and details.
*   **Enhanced Visual Language Understanding:** Leverages advanced Large Language Models (LLMs) for superior comprehension.
*   **Two-Stage Training:** Utilizes a two-stage approach to refine visual and language alignment, resulting in high-quality image-text generation.
*   **Efficient Fine-Tuning:** Achieves significant improvements in a computationally efficient manner, using a small, high-quality dataset.
*   **Open Source & Accessible:**  Built upon open-source projects like BLIP-2, Lavis, and Vicuna, making it accessible for research and experimentation.
*   **Multiple Deployment Options:**  Available through a web demo, Hugging Face Spaces, and Colab notebooks.

## Online Demo

Interact with MiniGPT-4 directly by uploading images and chatting.

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples on the [Project Page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---
## What's New

*   **Vicuna-Aligned MiniGPT-4:** A pre-trained MiniGPT-4 is now available, aligned with Vicuna-7B, reducing GPU memory consumption down to as low as 12GB.

---

## Installation Guide

### 1. Set up the Environment

Clone the repository, create a Python environment, and activate it using the following commands:

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

### 2. Prepare Vicuna Weights

**(Alternative: Download Pre-Prepared Weights - check the original README for details.)**

MiniGPT-4 is built upon Vicuna-13B v0.  Prepare Vicuna weights following the instructions [here](PrepareVicuna.md) or the steps outlined below.

To prepare Vicuna weights, download the delta weights from: https://huggingface.co/lmsys/vicuna-13b-delta-v1.1

```bash
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v1.1
```

You'll also need the original LLaMA-13B weights.  See the original README for links. Consider using methods like downloading from IPFS or using a torrent, if direct download is not possible.

```
# Full backup: ipfs://Qmb9y5GCkTG7ZzbBWMu2BXwMkzyCKcUjtEKPpgdZ7GEFKm

# Follow the instructions within the original README for details on installing the models
```
Download the required files. Then use this script to convert them to a Hugging Face format:

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

After converting to the Hugging Face format, you'll create the final working weights.

```bash
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10

python -m fastchat.model.apply_delta --base /path/to/llama-13b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-13b-delta-v0/
```

Finally, set the path to your Vicuna weights in the model configuration file [minigpt4/configs/models/minigpt4.yaml#L16].

### 3. Prepare MiniGPT-4 Checkpoints

Download the pre-trained checkpoint based on your Vicuna model (13B or 7B).

|                                Checkpoint Aligned with Vicuna 13B                                |                               Checkpoint Aligned with Vicuna 7B                                |
:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
 [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link) | [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

Set the path to the pre-trained checkpoint in the evaluation configuration file [eval\_configs/minigpt4\_eval.yaml#L10].

### 4. Launch the Demo Locally

Run the demo on your local machine with:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

---
## Training

MiniGPT-4 is trained in two alignment stages.

**1. Stage 1 Pretraining**

Train the model on image-text pairs.
Refer to `dataset/README_1_STAGE.md` for data preparation.
Run the first stage training with:

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

Download the MiniGPT-4 checkpoint from [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2. Stage 2 Fine-tuning**

Fine-tune MiniGPT-4 with a high-quality image-text dataset.  Refer to `dataset/README_2_STAGE.md` for dataset preparation. Specify the checkpoint path from Stage 1 in `train_configs/minigpt4_stage2_finetune.yaml`.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

---
## Acknowledgements

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

---
## Citation

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

---
## Community

*   Join the community groups (see the original README).

---
## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md).
Code is based on [Lavis](https://github.com/salesforce/LAVIS), with the license [here](LICENSE_Lavis.md).