# MiniGPT-4: Unlock Visual Language Understanding with Advanced Large Language Models

**MiniGPT-4 revolutionizes how we understand images by seamlessly integrating visual and textual information, enabling powerful new applications.**  Developed by Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny from King Abdullah University of Science and Technology, this project leverages cutting-edge large language models to provide enhanced visual language understanding.

[Link to Original Repo:](https://github.com/RiseInRose/MiniGPT-4-ZH)

**Key Features:**

*   **Enhanced Visual Language Understanding:** MiniGPT-4 excels at bridging the gap between images and text, offering a deeper understanding of visual content.
*   **Two-Stage Training:** Employs a two-stage training process for optimal performance, including a pre-training phase with 5 million image-text pairs and a fine-tuning phase with a high-quality, curated dataset.
*   **Integration of BLIP-2 and Vicuna:** Leverages BLIP-2 for visual encoding and the powerful Vicuna LLM for robust language generation.
*   **High-Quality Data Generation:** Employs a novel method to generate high-quality image-text pairs using the model and ChatGPT, leading to improved performance.
*   **Open-Source and Accessible:** This project is built on open-source libraries and offers a user-friendly interface, including a Colab notebook and a Hugging Face Space demo.

## Quick Start

### 1. Environment Setup

*   Clone the repository:
    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    ```
*   Create and activate a Python environment:
    ```bash
    conda env create -f environment.yml
    conda activate minigpt4
    ```

### 2. Prepare Vicuna Weights

*   **Download Weights:**  (Refer to the original README for detailed instructions on acquiring the Vicuna-13B delta weights and the original LLaMA-13B weights.)

### 3. Prepare MiniGPT-4 Checkpoints

*   **Download Checkpoints:** Download pre-trained checkpoints aligned with your chosen Vicuna version (13B or 7B) from the provided links.

    *   Checkpoints aligned with Vicuna 13B: [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
    *   Checkpoints aligned with Vicuna 7B: [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

### 4. Launch the Demo

*   Run the demo locally:

    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
    ```

## Resources

*   **Online Demo:** [Click here](https://minigpt-4.github.io) to interact with MiniGPT-4 directly.

*   **Project Page:** [Visit the project page](https://minigpt-4.github.io) for more examples and details.

*   **Hugging Face Spaces:**
    *   [Demo on Hugging Face Spaces](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
    *   [MiniGPT-4 Model on Hugging Face](https://huggingface.co/Vision-CAIR/MiniGPT-4)

*   **Colab Notebook:** [Get started with a Colab notebook](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)

*   **YouTube Video:** [Watch the MiniGPT-4 introduction](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## Training

MiniGPT-4's training involves two key alignment stages:

**1. Stage 1: Pre-training**

*   Utilize a dataset of image-text pairs from Laion and CC datasets to align vision and language models.  (See dataset/README\_1\_STAGE.md for dataset preparation instructions).
*   Run the training command:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

*   First-stage checkpoints are available [here](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link).

**2. Stage 2: Fine-tuning**

*   Use a curated dataset of image-text pairs in a dialogue format for further alignment. (See dataset/README\_2\_STAGE.md for dataset preparation instructions).
*   Specify the path to the Stage 1 checkpoint in `train_configs/minigpt4_stage2_finetune.yaml`.
*   Run the training command:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Acknowledgements

*   BLIP2
*   Lavis
*   Vicuna

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md). Many parts of the code are based on [Lavis](https://github.com/salesforce/LAVIS) licensed under the [BSD 3-Clause](LICENSE_Lavis.md).

## Community

[Join the AI commercial application exchange group](insert WeChat QR code image) and the knowledge planet (insert a WeChat QR code image)