# MiniGPT-4: Unleash the Power of Visual Language Understanding (Enhanced & Simplified)

**Explore the world with words! MiniGPT-4 brings cutting-edge visual language understanding to your fingertips.** This repository, developed by Zhu Deyao, Chen Jun, Shen Xiaoqian, Li Xiang, and Mohamed Elhoseiny from King Abdullah University of Science and Technology, leverages advanced large language models to create a powerful AI capable of interpreting and describing images.  [**View the original repository on GitHub**](https://github.com/RiseInRose/MiniGPT-4-ZH).

## Key Features

*   **Image-to-Text Generation:**  Describe images in detail, providing insightful captions and explanations.
*   **Visual Question Answering:** Get answers to your questions about the content of an image.
*   **Interactive Dialogue:**  Engage in conversations about images, exploring their features and context.
*   **Open Source & Accessible:** Built upon open-source foundations like BLIP-2 and Vicuna, making it easier to experiment and contribute.
*   **Two-Stage Training:** Employs a two-stage training process with a curated dataset to enhance the model's performance and usability.

## Online Demo

Interact directly with MiniGPT-4 to see it in action! Upload an image and start a conversation to understand its contents.

[![Demo](figs/online_demo.png)](https://minigpt-4.github.io)

More examples and information are available on the [Project Page](https://minigpt-4.github.io).

[<img src='https://img.shields.io/badge/Project-Page-Green'>](https://minigpt-4.github.io)
[<img src='https://img.shields.io/badge/Paper-PDF-red'>](MiniGPT_4.pdf)
[<img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'>](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
[<img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'>](https://huggingface.co/Vision-CAIR/MiniGPT-4)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## News and Updates
*   **MiniGPT-4 with Vicuna-7B:** We now offer a pre-trained MiniGPT-4 aligned with Vicuna-7B! This version requires as little as 12GB of GPU memory for the demo.
## Getting Started

### Installation
1.  **Clone the Repository and Set Up Environment:**
```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

2.  **Prepare the Vicuna Weights:**  Follow the instructions in [PrepareVicuna.md](PrepareVicuna.md) to prepare the required Vicuna weights.  Alternatively, you can download pre-prepared weights (check the original README for potential restrictions.)

3.  **Prepare the MiniGPT-4 Checkpoint:**
    *   Download the pre-trained checkpoint aligned with your Vicuna model version from the provided links (see original README).

### Run the Demo Locally

Run the following command to experience the demo on your machine:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```
## Training
MiniGPT-4 is trained in two stages:

1.  **Stage 1: Pre-training:**  Visual and language models are aligned using image-text pairs from Laion and CC datasets.

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

2.  **Stage 2: Fine-tuning:** The model is fine-tuned on a smaller, high-quality dataset converted into a dialogue format.

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Acknowledgements

*   **BLIP-2:** MiniGPT-4 utilizes the BLIP-2 architecture.
*   **LAVIS:**  The repository is built upon LAVIS.
*   **Vicuna:**  The impressive capabilities of the open-source Vicuna LLM.

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

This project is licensed under the [BSD 3-Clause License](LICENSE.md).  Code is also based on [Lavis](https://github.com/salesforce/LAVIS), which is licensed under the BSD 3-Clause License [here](LICENSE_Lavis.md).

##  (Original) Contributions

This project is forked from [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), and large parts of the content are based on the original work.