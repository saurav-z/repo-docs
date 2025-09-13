## MiniGPT-4: Unleash the Power of Visual Language Understanding

**MiniGPT-4 empowers you to understand images by leveraging advanced large language models, offering a new dimension in visual AI.** For the original repository, visit [RiseInRose's MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH).

**Key Features:**

*   **Visual-Language Understanding:** Analyze images and generate descriptive text.
*   **Two-Stage Training:**  Utilizes a pre-training phase and a fine-tuning stage for enhanced performance.
*   **User-Friendly:** Easily chat with MiniGPT-4 to explore image content.
*   **Open Source:** Built upon the foundation of BLIP-2, Lavis, and Vicuna for accessibility and innovation.
*   **Efficient Resource Usage:** Optimizations enable operation on systems with limited GPU memory.

**Online Demo:**

Interact with MiniGPT-4 directly! Click the image below to start a conversation about your uploaded image:

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

Explore more examples on the [Project Page](https://minigpt-4.github.io).

<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/minigpt4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a> <a href='https://huggingface.co/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

**Key Updates:**

*   **Pre-trained MiniGPT-4 with Vicuna-7B:** Now offers a version that requires as little as 12GB of GPU memory for the demo.
*   **Community Support:**  Resources and guidance are available from various community members for deployment and troubleshooting.

---

**Getting Started:**

### Installation

**1. Code and Environment Setup:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    ```

2.  Create and activate a Python environment:

    ```bash
    conda env create -f environment.yml
    conda activate minigpt4
    ```

**2. Prepare Vicuna Weights:**

*   Detailed instructions on preparing the Vicuna weights, including downloading and conversion, are provided [here](PrepareVicuna.md).
    *  **Simplified Steps (For those who don't want to do all the hard work):**
        *   Download the LLaMA-13B weights from [here](https://github.com/facebookresearch/llama/issues/149)
        *   Follow the documentation to prepare the weights.
*   For convenience, a pre-trained weight package may be available from the contributor.

**3. Prepare Pre-trained MiniGPT-4 Checkpoints:**

*   Choose a checkpoint based on your Vicuna model version (13B or 7B) and set the path in `eval_configs/minigpt4_eval.yaml`.
    *   Vicuna 13B: [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
    *   Vicuna 7B: [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

### Running the Demo Locally

Run the demo using:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

*   The demo runs in 8-bit to save GPU memory. You can adjust settings in the config for more robust GPU setups.

### Training

**1.  Stage 1: Pre-training**

*   Pre-train the model using image-text pairs.
*   Prepare your data and consult [dataset/README_1_STAGE.md].
*   Run training:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

*   A checkpoint from this stage is available [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2.  Stage 2: Fine-tuning**

*   Fine-tune on a smaller, high-quality image-text dataset with a dialog format.
*   Prepare your data and consult [dataset/README_2_STAGE.md].
*   Specify the Stage 1 checkpoint path in  `train_configs/minigpt4_stage2_finetune.yaml`.
*   Run training:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

---

**Acknowledgments:**

*   [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [Lavis](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

**Citation:**

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

---

**Join the Community:**

*   Explore online versions and stay informed about AI developments.

|              Join via WeChat               |                Knowledge Sharing (Chinese)                 |
|:-------------------------------:|:-----------------------------------------------:|
| <img src="./img/qrcode.png" width="300"/> |  <img src="./img/WechatIMG81.jpeg" width="300"/> |

---

**License:**

*   This project is licensed under the [BSD 3-Clause License](LICENSE.md).
*   The code is based on [Lavis](https://github.com/salesforce/LAVIS), which is licensed under the BSD 3-Clause License [here](LICENSE_Lavis.md).

---

**Contributions:**

This project is forked from [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and this documentation is largely based on the original.