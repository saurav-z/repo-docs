# MiniGPT-4: Unleash Visual Language Understanding with Advanced LLMs

**Experience the power of visual language!** MiniGPT-4, developed by Zhu Deyao, Chen Jun, Shen Xiaoqian, Li Xiang, and Mohamed Elhoseiny from King Abdullah University of Science and Technology, uses cutting-edge large language models to bridge the gap between vision and language. This repository provides resources for understanding, implementing, and experimenting with MiniGPT-4.  For more information, visit the original repository: [https://github.com/RiseInRose/MiniGPT-4-ZH](https://github.com/RiseInRose/MiniGPT-4-ZH)

## Key Features:

*   **Enhanced Visual Language Understanding:** Leverages the power of advanced LLMs like Vicuna to interpret and generate text based on images.
*   **Two-Stage Training:** Employs a two-stage training process, including pre-training and fine-tuning, to achieve superior performance.
*   **High-Quality Dataset Creation:** Utilizes a novel approach to create a high-quality image-text dataset, leading to improved generation quality.
*   **Efficient Fine-Tuning:** The second fine-tuning stage is computationally efficient, requiring minimal resources.
*   **Open-Source and Accessible:** Provides open-source code and resources for easy deployment and experimentation.

## Quick Start

### Online Demo

Explore MiniGPT-4's capabilities instantly:

[![demo](figs/online_demo.png)](https://minigpt-4.github.io)

### Installation

Here's a streamlined guide to get you started:

**1. Set up the Environment:**

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```

**2. Prepare the Vicuna Weights**
Detailed instructions for preparing the Vicuna weights can be found in the original repository. Due to potential copyright issues, the model weights are not hosted here.  Follow the instructions to download the model weights and convert them to the Hugging Face Transformers format.  A helpful guide is available at [PrepareVicuna.md](PrepareVicuna.md) in the original repo.

**3. Prepare the MiniGPT-4 Checkpoint**

Download the pre-trained checkpoints based on the Vicuna model you prepared:

*   **Vicuna 13B:** [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)
*   **Vicuna 7B:** [Download](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)

Then, set the checkpoint path in the evaluation configuration file (e.g., `eval_configs/minigpt4_eval.yaml` at line 11).

**4. Run the Demo Locally**

Launch the local demo with the following command:

```bash
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

### Training

The training process involves two stages.  Full instructions are in the original README, this section just provides a summary.

**1. Stage 1: Pretraining**

Train the model using image-text pairs from datasets like Laion and CC.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

**2. Stage 2: Fine-tuning**

Fine-tune the model on a high-quality image-text dataset.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Resources

*   **Project Page:** [https://minigpt-4.github.io](https://minigpt-4.github.io)
*   **Paper (PDF):** [MiniGPT_4.pdf](MiniGPT_4.pdf)
*   **Hugging Face Spaces:**  [Vision-CAIR/minigpt4](https://huggingface.co/spaces/Vision-CAIR/minigpt4)
*   **Hugging Face Model:** [Vision-CAIR/MiniGPT-4](https://huggingface.co/Vision-CAIR/MiniGPT-4)
*   **Colab Notebook:** [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OK4kYsZphwt5DXchKkzMBjYF6jnkqh4R?usp=sharing)
*   **YouTube:** [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=__tftoxpBAw&feature=youtu.be)

## Acknowledgements

*   **BLIP-2:** Model architecture inspiration.
*   **Lavis:** The repository is built upon Lavis!
*   **Vicuna:** For the amazing language capabilities.

## Citation

If you use MiniGPT-4 in your research or applications, please cite the following:

```bibtex
@misc{zhu2022minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and xiang Li and Mohamed Elhoseiny},
      year={2023},
}
```

## License

This repository is under the [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS), which is under the [BSD 3-Clause License](LICENSE_Lavis.md).