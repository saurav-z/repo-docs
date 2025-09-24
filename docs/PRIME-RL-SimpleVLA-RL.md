<div align="center">
  <a href="https://github.com/PRIME-RL/SimpleVLA-RL">
    <img src="figs/logo.png" width="260" alt="SimpleVLA-RL Logo"/>
  </a>
</div>

## SimpleVLA-RL: Revolutionizing VLA Training with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674)
[![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL)
[![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804)
[![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

**SimpleVLA-RL leverages simple 0/1 rewards to enable effective online reinforcement learning (RL) for Vision-Language-Action (VLA) models, achieving state-of-the-art performance.**

**Key Features:**

*   **Superior Performance:** Achieves state-of-the-art results on the LIBERO benchmark, significantly improving upon existing methods.
*   **Data-Efficient Learning:** Demonstrates significant performance gains with limited data, crucial for data-scarce environments.
*   **Novel Insights:** Reveals a "pushcut" new-action phenomenon, enhancing the understanding of RL in VLA models.
*   **Enhanced Generalization:** Improves spatial, object, and goal generalization capabilities in VLA models.
*   **Easy to use**: Integrates seamlessly with popular VLA models like OpenVLA-OFT.

<div align="center">
  <img src="figs/teaser.png" alt="Overview of SimpleVLA-RL." width="90%" />
  <figcaption><em>Overview of SimpleVLA-RL: An efficient RL framework for VLA, boosting long-horizon planning, outperforming SFT, and enhancing generalization.</em></figcaption>
</div>

## Table of Contents

*   [News](#news)
*   [Overview](#overview)
*   [Main Results](#main-results)
*   [Getting Started](#getting-started)
    *   [1. Set Up the Environment](#1-set-up-the-environment)
    *   [2. Prepare the SFT Model](#2-prepare-the-sft-model)
    *   [3. Train with SimpleVLA-RL](#3-train-with-simplevla-rl)
    *   [4. Run Evaluation](#4-run-evaluation)
*   [Acknowledgement](#acknowledgement)
*   [Contact](#contact)
*   [TODO](#todo)
*   [Citation](#citation)

## News

*   **[2025-09-12]** Paper Release: SimpleVLA-RL paper is now available! [Read the Paper](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** Code Release: SimpleVLA-RL code is now available for experimentation.

## Overview

SimpleVLA-RL is a straightforward yet highly effective approach for online Reinforcement Learning (RL) specifically designed for Vision-Language-Action (VLA) models. It harnesses the power of outcome-level 0/1 rule-based reward signals, directly derived from simulation environments, to enhance VLA model training.

<div align="center">
  <img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

## Main Results

SimpleVLA-RL achieves outstanding results when evaluated on the LIBERO benchmark using OpenVLA-OFT, pushing performance to **97.6 points** on LIBERO-Long and establishing a new state-of-the-art. Furthermore, with only one trajectory per task for cold-start Supervised Fine-Tuning (SFT), SimpleVLA-RL significantly improves the performance of OpenVLA-OFT from 17.3 to 91.7, showcasing an impressive **74.4-point (430.1%) improvement**.

<div align="center">
  <img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

## Getting Started

Follow these steps to get started with SimpleVLA-RL.

### 1. Set Up the Environment

This project is built upon [veRL](https://verl.readthedocs.io/en/latest/start/install.html). Install veRL and the necessary environment for your Vision-Language-Action (VLA) model. Instructions for setting up **OpenVLA-OFT** are below:

*   **Install veRL**
    Follow the official veRL installation guide: [veRL Installation](https://verl.readthedocs.io/en/latest/start/install.html).

*   **Install OpenVLA-OFT**
    Set up OpenVLA-OFT by following the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft).

### 2. Prepare the SFT Model

You'll need an SFT (Supervised Fine-Tuning) VLA model for RL training. Choose from the following options:

*   **OpenVLA-OFT SFT Models**
    Download from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Available models include:
    *   `libero-10 traj1 SFT`
    *   `libero-10 trajall SFT`
*   **OpenVLA SFT Models**
    Download from [here](https://huggingface.co/openvla).
*   **Other Models**
    Fine-tune other models as needed.

### 3. Train with SimpleVLA-RL

Configure the following before running the training script:

*   **WandB API Key**
    Replace the `WANDB_API_KEY` in `SimpleVLA-RL/align.json` with your WandB API key.
*   **Key Variables**
    Update the following in `examples/run_openvla_oft_rl.sh`:
    *   `WANDB_API_KEY`: Your WandB API key.
    *   `EXPERIMENT_NAME`: Experiment name.
    *   `SFT_MODEL_PATH`: SFT model path.
    *   `CKPT_PATH`: Checkpoint save path.
    *   `DATASET_NAME`: Options: `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`.
    *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
    *   `NUM_GPUS`: GPUs per node (e.g., `8`).
    *   `NUM_NODES`: Number of nodes (e.g., `1`).

>   [!NOTE]
>   The script has been tested with:  
>   *   Single-node: `NUM_NODES=1`, `NUM_GPUS=8` (1 node with 8 NVIDIA A800 GPUs, each 80GB memory).
>   *   Multi-node: `NUM_NODES=2`, `NUM_GPUS=8` (2 nodes with 16 NVIDIA A800 GPUs, each 80GB memory).
>   *   Driver version: `470.161.03`, CUDA version: `12.4`.

*   **Run RL Training**
    Execute the following command to start RL training for OpenVLA-OFT on the LIBERO benchmark:

    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

### 4. Run Evaluation

To evaluate your model, set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and run the script:

```bash
bash examples/run_openvla_oft_rl.sh
```

## Acknowledgement

This project is built upon the great work of [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME). We are grateful for their contributions!

## Contact

*   Haozhan Li: zhan72426@gmail.com
*   Ning Ding: dingning@mail.tsinghua.edu.cn

## TODO

*   **Models**:
    *   ✅ Support OpenVLA and OpenVLA-OFT
    *   ⏳ Support Pi0 fast tokenizer
*   **Benchmarks**:
    *   ✅ Support LIBERO benchmark
    *   ⏳ Support RoboTwin benchmark

## Citation

If you find SimpleVLA-RL useful, please cite:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```

For more details, please visit the [SimpleVLA-RL repository](https://github.com/PRIME-RL/SimpleVLA-RL).