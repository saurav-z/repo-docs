## MiniGPT-4: Unleashing Visual Language Understanding with Large Language Models

**MiniGPT-4 allows you to chat with images, offering an advanced understanding of visual content through the power of large language models.** Check out the original repository [here](https://github.com/RiseInRose/MiniGPT-4-ZH).

**Key Features:**

*   **Image-Based Dialogue:** Engage in conversations about images, gaining insights into their content and context.
*   **Two-Stage Training:** Leverages a two-stage training process for enhanced performance and usability.
*   **High-Quality Dataset:** Employs a carefully curated dataset of image-text pairs for improved generation quality.
*   **Efficient Fine-tuning:** Utilizes an efficient fine-tuning process, requiring minimal computational resources.
*   **Emerging Capabilities:** Demonstrates advanced visual language abilities similar to those found in GPT-4.

**Get Started:**

*   **Online Demo:** Explore the capabilities of MiniGPT-4 by interacting with the [online demo](https://minigpt-4.github.io) to understand your images.
    [![demo](figs/online_demo.png)](https://minigpt-4.github.io)
*   **Project Resources:** Discover more examples and details on the [project page](https://minigpt-4.github.io).

    [![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://minigpt-4.github.io)
    [![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](MiniGPT_4.pdf)
    [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
    [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Vision-CAIR/MiniGPT-4)
    [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
    [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

**Installation**
(condensed from the original README, including only the most important steps, and linking out for details where possible)

1.  **Clone and Setup Environment**:
    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    conda env create -f environment.yml
    conda activate minigpt4
    ```
2.  **Prepare Vicuna Weights**:
    *   Download or prepare the Vicuna-13B v1.1 weights. Detailed instructions are in the original README, and also [here](PrepareVicuna.md)
    *   Download and prepare LLaMA 13B, if using the method for the delta weights.  Details [here](https://github.com/facebookresearch/llama/issues/149).
    *   Convert LLaMA weights using:

        ```bash
        python src/transformers/models/llama/convert_llama_weights_to_hf.py \
            --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
        ```
        *See the original README for troubleshooting tips.
    *   Use FastChat tools to create the final weights.
        ```bash
        pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
        python -m fastchat.model.apply_delta --base /path/to/llama-13b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-13b-delta-v0/
        ```
    *   Point to the Vicuna weights in  `minigpt4/configs/models/minigpt4.yaml` (Line 16).

3.  **Prepare MiniGPT-4 Checkpoint**:
    *   Download the pre-trained checkpoint for your Vicuna model (7B or 13B)
    *   Specify the checkpoint path in `eval_configs/minigpt4_eval.yaml` (Line 11).

4.  **Run Local Demo**:
    ```bash
    python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
    ```

5.  **Training**
    *   Training in 2 Stages
        *   Stage 1:  Pretraining uses image-text pairs from Laion and CC datasets.  See `dataset/README_1_STAGE.md`.
            ```bash
            torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
            ```
        *   Stage 2: Fine-tuning with a custom dataset.  See `dataset/README_2_STAGE.md`.
            ```bash
            torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
            ```

**Additional Resources:**

*   **Windows Deployment:**  See [issue 28](https://github.com/Vision-CAIR/MiniGPT-4/issues/28) in the original repo.
*   **Colab Notebook:** [Link to Colab notebook](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)

**Acknowledgements:**

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

**License:**

*   [BSD 3-Clause License](LICENSE.md)
*   Lavis is also BSD 3-Clause licensed [here](LICENSE_Lavis.md).