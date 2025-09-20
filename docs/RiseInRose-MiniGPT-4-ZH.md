# MiniGPT-4: Unleashing Vision-Language Understanding with Large Language Models

**MiniGPT-4 leverages advanced large language models to enhance visual language understanding, allowing you to chat with images and explore their details.**  [Visit the original repository on GitHub](https://github.com/RiseInRose/MiniGPT-4-ZH).

**Authors:** Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny (King Abdullah University of Science and Technology)

## Key Features

*   **Visual-Language Alignment:** MiniGPT-4 aligns a frozen visual encoder from BLIP-2 with the frozen LLM Vicuna.
*   **Two-Stage Training:** The model undergoes pre-training and fine-tuning for enhanced performance and usability.
*   **High-Quality Dataset:** A curated dataset improves generation reliability and overall user experience.
*   **Emerging Capabilities:** MiniGPT-4 exhibits impressive vision-language capabilities similar to GPT-4.
*   **Open-Source & Accessible:**  Explore and experiment with cutting-edge AI models.

## Quick Start

### 1. Environment Setup
*   Clone the repository:
    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    ```
*   Create and activate a Conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate minigpt4
    ```

### 2. Get the Pre-trained Vicuna Weights

*   Download pre-trained weights (links/instructions can be found in the original README).
*   Detailed instructions are available [here](PrepareVicuna.md).  This involves downloading LLaMA weights (links in original README), then delta weights from  https://huggingface.co/lmsys/vicuna-13b-delta-v1.1.
*   Convert the weights using the provided scripts and instructions.

    ```bash
    git clone https://github.com/lm-sys/FastChat
    cd FastChat
    # 查看tag
    git tag
    # 切换到最新的tag分支
    git checkout v0.2.3
    # 安装
    pip install e .

    # 安装其他依赖
    pip install transformers[sentencepiece]
    ```

### 3. Prepare the MiniGPT-4 Checkpoint

*   Download the pre-trained checkpoint aligned with your chosen Vicuna version (13B or 7B) from the links provided in the original README.

### 4. Run the Demo

*   Run the demo locally using:
    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    ```

## Further Resources

*   **Online Demo:** [Interact with the demo](https://minigpt-4.github.io) to explore its capabilities.
*   **Project Page:** [Visit the project page](https://minigpt-4.github.io) for more examples.
*   **Hugging Face:** [Model](https://huggingface.co/Vision-CAIR/MiniGPT-4) and [Spaces](https://huggingface.co/spaces/Vision-CAIR/minigpt4).
*   **Colab:** [Colab Notebook](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
*   **YouTube:** [YouTube Video](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## Training

*   **Stage 1: Pre-training:** Train the model on image-text pairs from Laion and CC datasets.  Follow the [first stage dataset preparation](dataset/README_1_STAGE.md).  Run training with:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```

*   **Stage 2: Fine-tuning:** Fine-tune the model with a curated dataset in a dialogue format. Follow the [second stage dataset preparation](dataset/README_2_STAGE.md). Run fine-tuning with:

    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

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

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md). The code is based on [Lavis](https://github.com/salesforce/LAVIS) (BSD 3-Clause License: [LICENSE_Lavis.md]).