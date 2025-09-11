# MiniGPT-4: Unleash Vision-Language Understanding with Large Language Models

**MiniGPT-4 is an innovative model that allows you to chat with images and explore their details, thanks to advanced large language models.**  For the original repository, visit: [https://github.com/RiseInRose/MiniGPT-4](https://github.com/RiseInRose/MiniGPT-4).

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny.

**Affiliation:** King Abdullah University of Science and Technology (KAUST)

## Key Features:

*   **Visual-Language Understanding:** MiniGPT-4 leverages a frozen visual encoder from BLIP-2 and a frozen LLM (Vicuna) for enhanced image understanding.
*   **Two-Stage Training:** The model undergoes a two-stage training process: a pre-training phase and a fine-tuning phase.
*   **High-Quality Dataset Generation:** Employs a novel method to create high-quality image-text pairs, leading to improved performance.
*   **Efficient Fine-tuning:** The fine-tuning phase is computationally efficient, requiring minimal resources.
*   **Emerging Capabilities:** MiniGPT-4 demonstrates capabilities similar to those found in GPT-4, demonstrating an ability to discuss image content in a natural way.

## Online Demo:

Interact with MiniGPT-4 directly to learn more about your images.

[![MiniGPT-4 Demo](figs/online_demo.png)](https://minigpt-4.github.io)

## Resources:

*   **Project Page:** [https://minigpt-4.github.io](https://minigpt-4.github.io)
*   **Paper (PDF):** [MiniGPT_4.pdf](MiniGPT_4.pdf)
*   **Hugging Face Spaces:** [Vision-CAIR/minigpt4](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
*   **Hugging Face Model:** [Vision-CAIR/MiniGPT-4](https://huggingface.co/Vision-CAIR/MiniGPT-4)
*   **Google Colab:** [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
*   **YouTube Demo:** [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News

*   A pre-trained MiniGPT-4 aligned with Vicuna-7B is now available! The demo requires as little as 12GB of GPU memory.

## Getting Started:

### Installation:

**1. Clone the repository and create a Conda environment:**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare the Vicuna weights.**

*   For the specific instructions on how to download, convert, and prepare the weights, please refer to the file [PrepareVicuna.md](PrepareVicuna.md) in the original repository.  This involves acquiring the LLaMA weights, the Vicuna delta weights and converting these to the Hugging Face Transformers format.

**3. Prepare the MiniGPT-4 checkpoint:**

*   Download the pre-trained checkpoint according to the Vicuna model you are using. Links can be found in the original README.

### Run the Demo Locally:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

## Training:

MiniGPT-4 training consists of two alignment stages.

**1. Stage 1: Pre-training**

*   Train the model using image-text pairs from Laion and CC datasets.
*   See [dataset/README_1_STAGE.md](dataset/README_1_STAGE.md) for dataset preparation.
*   Start training:
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

**2. Stage 2: Fine-tuning**

*   Use a small, high-quality image-text dataset, converted to a dialogue format.
*   See [dataset/README_2_STAGE.md](dataset/README_2_STAGE.md) for dataset preparation.
*   Update `train_configs/minigpt4_stage2_finetune.yaml` with the checkpoint path from Stage 1.
*   Start fine-tuning:
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Acknowledgements:

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [LAVIS](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

## Citation:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models}, 
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License:

*   This repository is licensed under the [BSD 3-Clause License](LICENSE.md).
*   The code is based on [Lavis](https://github.com/salesforce/LAVIS), licensed under [BSD 3-Clause](LICENSE_Lavis.md).