# MiniGPT-4: Unleash Vision-Language Understanding with Advanced Large Language Models

**Experience the power of MiniGPT-4, a cutting-edge model that bridges the gap between vision and language, allowing you to have insightful conversations about images.** Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology, and based on the original work from [Vision-CAIR](https://github.com/Vision-CAIR/MiniGPT-4).

*   **Key Features:**
    *   **Advanced Vision-Language Understanding:** MiniGPT-4 leverages a frozen visual encoder from BLIP-2 and aligns it with the powerful Vicuna LLM, enabling a deep understanding of visual content.
    *   **Two-Stage Training:** The model undergoes two key training stages: pre-training with a large image-text dataset and fine-tuning with a smaller, high-quality dataset generated with the model itself and ChatGPT.
    *   **High-Quality Image-Text Generation:** MiniGPT-4 excels at generating comprehensive and reliable descriptions and analyses of images, similar to capabilities seen in GPT-4.
    *   **Efficient Fine-Tuning:** The second fine-tuning stage is computationally efficient, requiring only a single A100 GPU for just 7 minutes, dramatically improving the model's usability.
    *   **Easy to Deploy and Test:** Includes straightforward installation instructions and a demo for quick evaluation.

*   [Project Page](https://minigpt-4.github.io)
*   [Paper PDF](MiniGPT_4.pdf)
*   [Hugging Face Spaces - Demo](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
*   [Hugging Face Model](https://huggingface.co/Vision-CAIR/MiniGPT-4)
*   [Colab Demo](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
*   [YouTube Demo](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

---

## Quick Start

### Installation

Follow these steps to get started:

1.  **Clone the Repository and Set Up Environment:**

    ```bash
    git clone https://github.com/RiseInRose/MiniGPT-4-ZH
    cd MiniGPT-4-ZH # Or whatever you named your local folder
    conda env create -f environment.yml
    conda activate minigpt4
    ```

2.  **Prepare Vicuna Weights:**

    *   Download pre-trained weights (availability details in the original README) or prepare them as described [in the original repository](https://github.com/Vision-CAIR/MiniGPT-4#2-prepare-pretrained-vicuna-weights).  This requires obtaining the base LLaMA-13B weights, applying delta weights, and converting the format.

    *   **Important Note:**  This repository is a Chinese-translated version of the original.  The original README contains detailed instructions regarding the preparation of Vicuna weights, including links to necessary files and instructions for downloading.  Refer to the original README for these details.

3.  **Prepare Pre-trained MiniGPT-4 Checkpoint:**

    *   Download the pre-trained checkpoint based on the Vicuna model (13B or 7B).  Links provided in the original README.

### Run the Demo

1.  **Local Demo:**  Run the demo.py script:

    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    ```

    *   Adjust GPU memory usage settings in `eval_configs/minigpt4_eval.yaml` and by setting `load_in_8bit=False` in `minigpt4/models/mini_gpt4.py` if you have a higher-spec GPU.

### Training

The training process consists of two stages, as described in the original README.

1.  **Stage 1 Pre-training:**  Use the first stage dataset to train the model.
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

2.  **Stage 2 Fine-tuning:**  Fine-tune using a smaller high-quality dataset.
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

---

## Acknowledgements

*   [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
*   [LAVIS](https://github.com/salesforce/LAVIS)
*   [Vicuna](https://github.com/lm-sys/FastChat)

---

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md).

---