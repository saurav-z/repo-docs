# MiniGPT-4: Revolutionizing Vision-Language Understanding with LLMs

**Unlock the power of visual understanding!** MiniGPT-4, developed by Zhu Deyao et al. from King Abdullah University of Science and Technology, combines a frozen visual encoder from BLIP-2 with the powerful Vicuna Large Language Model (LLM) to deliver cutting-edge vision-language capabilities. ([Original Repo](https://github.com/RiseInRose/MiniGPT-4-ZH))

## Key Features

*   **Enhanced Vision-Language Understanding:** Integrates visual and language models for deeper image analysis.
*   **Two-Stage Training:** Employs a two-stage training process for optimal performance and usability.
*   **High-Quality Dataset:** Leverages a curated dataset to improve the reliability of generated text.
*   **Efficient Fine-tuning:** Achieves significant improvements with minimal computational resources.
*   **Emerging Capabilities:** Demonstrates impressive visual language abilities similar to GPT-4.
*   **Open Source:** Leveraging the power of open-source LLMs like Vicuna

## Online Demo

Interact with MiniGPT-4 directly! Upload an image and explore its capabilities:

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

## Getting Started

### 1. Installation

**Prerequisites:**

*   Python environment (e.g., using `conda`)

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Vision-CAIR/MiniGPT-4.git
    cd MiniGPT-4
    ```
2.  **Create and activate a Python environment:**
    ```bash
    conda env create -f environment.yml
    conda activate minigpt4
    ```

### 2. Prepare Vicuna Weights

The current version of MiniGPT-4 is built upon Vicuna-13B v0. You can download the pre-trained weights (details in the original README) or refer to `PrepareVicuna.md` in the original repo for detailed instructions.

### 3. Prepare MiniGPT-4 Checkpoints

*   Download the pre-trained checkpoints based on the Vicuna model you've prepared. Download links are available in the original README (Google Drive links).
*   Set the checkpoint path in `eval_configs/minigpt4_eval.yaml` (line 11).

### 4. Run the Demo Locally

Run the demonstration script:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
```

## Training

MiniGPT-4 is trained in two stages:

1.  **Stage 1: Pretraining:** Align visual and language models using image-text pairs from Laion and CC datasets.  Details of dataset preparation in `dataset/README_1_STAGE.md`.  Run training with:
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
    ```
2.  **Stage 2: Fine-tuning:**  Fine-tune the model using a high-quality dataset of image-text pairs in a dialog format. Details of dataset preparation in `dataset/README_2_STAGE.md`.  Run training with:
    ```bash
    torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
    ```

## Acknowledgements

*   **BLIP-2:** For the model architecture.
*   **Lavis:** For the underlying framework.
*   **Vicuna:** For its amazing language capabilities.

## Citation

If you use MiniGPT-4 in your research, please cite:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This project is licensed under the [BSD 3-Clause License](LICENSE.md). Code based on [Lavis](https://github.com/salesforce/LAVIS) is licensed under the [BSD 3-Clause License](LICENSE_Lavis.md).