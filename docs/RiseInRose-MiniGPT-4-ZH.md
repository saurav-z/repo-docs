# MiniGPT-4: Unleash the Power of Visual Language Understanding (Simplified)

**MiniGPT-4 enhances visual language understanding by aligning a frozen vision encoder with a frozen large language model.**  This README provides a simplified guide to understanding and using MiniGPT-4, a powerful model for interacting with images.  For the original repository, please visit [RiseInRose's MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH).

**Key Features:**

*   **Image-to-Text Generation:** Describe images in detail, generating coherent and informative text.
*   **Interactive Dialogue:** Engage in conversations about images, asking questions and receiving insightful answers.
*   **Simplified Architecture:** Built upon BLIP-2 for visual encoding and Vicuna for language understanding.
*   **Two-Stage Training:** Uses a two-stage training process for alignment and improved usability.
*   **Open-Source & Accessible:** Based on open-source models like Vicuna, making it easier to use.

## Online Demo

Interact with the MiniGPT-4 demo to explore its capabilities.

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

More examples are available on the [Project Page](https://minigpt-4.github.io).

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://minigpt-4.github.io)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](MiniGPT_4.pdf)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Vision-CAIR/MiniGPT-4)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## Getting Started (Simplified)

### Installation

1.  **Set up the Environment:**

    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    conda env create -f environment.yml
    conda activate minigpt4
    ```

2.  **Prepare Vicuna Weights:** (Requires downloading weights)

    *   Follow the instructions to obtain the Vicuna-13B v1.1 weights.  You'll need to acquire the base LLaMA-13B weights and the delta weights from Hugging Face and apply the delta weights using `fastchat.model.apply_delta`.
    *   (Alternatively, the weights may be shared in the WeChat group or from the user's Colab instance)

3.  **Prepare MiniGPT-4 Checkpoint:**

    *   Download the pre-trained checkpoint aligned with your chosen Vicuna version (7B or 13B) from the provided Google Drive links.
    *   Set the path to the checkpoint in `eval_configs/minigpt4_eval.yaml` (line 11).

### Running the Demo

Run the demo locally:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

## Training (Overview)

MiniGPT-4 is trained in two stages.

1.  **Stage 1: Pretraining:**  Align visual and language models using image-text pairs from Laion and CC datasets.
2.  **Stage 2: Fine-tuning:** Further align MiniGPT-4 with a smaller, high-quality dataset created by the project to enhance usability.

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

## Community and Support

*   Join the WeChat group (see the QR code in the original README).

## License

*   This project uses the [BSD 3-Clause License](LICENSE.md).
*   Code based on [Lavis](https://github.com/salesforce/LAVIS) uses the [BSD 3-Clause License](LICENSE_Lavis.md).