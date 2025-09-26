# MiniGPT-4: Unleash Visual Language Understanding with Large Language Models

**MiniGPT-4 revolutionizes visual language understanding, enabling you to chat with images and unlock their hidden details.**  Explore this powerful model and its capabilities!  [Visit the original repository for more details](https://github.com/RiseInRose/MiniGPT-4-ZH).

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny (King Abdullah University of Science and Technology).

## Key Features

*   **Image-Based Conversation:**  Engage in interactive conversations about images, receiving detailed descriptions and insights.
*   **Advanced Visual Language Alignment:**  Leverages a projection layer to align a frozen visual encoder (BLIP-2) with a frozen large language model (Vicuna).
*   **Two-Stage Training:** Utilizes a two-stage training approach for optimal performance:
    *   **Stage 1: Pre-training:** Initial alignment using a large dataset of image-text pairs.
    *   **Stage 2: Fine-tuning:**  Refinement with a smaller, high-quality dataset, leading to improved reliability and usability.
*   **Efficient Training:** The second fine-tuning stage is computationally efficient, requiring only a single A100 GPU for approximately 7 minutes.
*   **Open-Source & Accessible:** Based on open-source foundations like BLIP-2, Lavis, and Vicuna, fostering collaboration and innovation.

## Demo & Resources

*   **Online Demo:**  [Try the interactive demo](https://minigpt-4.github.io) to experience MiniGPT-4's image understanding capabilities.
    [![demo](figs/online_demo.png)](https://minigpt-4.github.io)

*   **Project Page:**  Explore more examples and details on the [Project Page](https://minigpt-4.github.io).
    <a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

*   **Paper:**  Read the full research paper:  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
*   **Hugging Face Spaces:**  Experiment with the model on Hugging Face Spaces: <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>  and  <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
*   **Colab:**  Run the code in Google Colab: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
*   **YouTube:** Watch the demo: [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## Installation & Getting Started

Follow these steps to set up MiniGPT-4:

### 1. Code & Environment Setup

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

### 2. Prepare Vicuna Weights (LLM)

*   **Download Pre-trained Weights (Optional):** [Link to weights - potential access restriction due to licensing. Check the original repo or the community.]
*   **Prepare Vicuna Weights Manually (Recommended for advanced users):** Follow the instructions to prepare the Vicuna weights.  This involves:
    1.  Download the Vicuna delta weights from [https://huggingface.co/lmsys/vicuna-13b-delta-v1.1](https://huggingface.co/lmsys/vicuna-13b-delta-v1.1).
    2.  Obtain the original LLaMA-13B weights (details provided in the original repo - use at your own risk as it involves accessing unofficial sources).
    3.  Convert the LLaMA weights to Hugging Face Transformers format using a conversion script.
    4.  Use the `fastchat.model.apply_delta` script to apply the delta weights to the base LLaMA weights.
*   **Configure the path in  `minigpt4/configs/models/minigpt4.yaml`**

### 3. Prepare MiniGPT-4 Checkpoint

*   **Download the pre-trained checkpoint** based on the Vicuna version you are using (13B or 7B).
    *   **Vicuna 13B:** [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
    *   **Vicuna 7B:** [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)
*   **Configure the path in `eval_configs/minigpt4_eval.yaml`**

### 4. Run the Demo Locally

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

## Training

### 1. Stage 1: Pre-training

Follow the instructions in `dataset/README_1_STAGE.md` to prepare the dataset.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

Download the checkpoint of stage 1 pretrain: [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link)

### 2. Stage 2: Fine-tuning

Follow the instructions in `dataset/README_2_STAGE.md` to prepare the dataset.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Acknowledgements

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
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

This project is licensed under the [BSD 3-Clause License](LICENSE.md).  Code based on [Lavis](https://github.com/salesforce/LAVIS) is licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).

## Further Information and Community

[Please refer to the original GitHub repo for detailed instructions and the latest updates.](https://github.com/RiseInRose/MiniGPT-4-ZH)