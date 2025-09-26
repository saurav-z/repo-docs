<div align="center">

<img src="figs/logo.png" width="260" alt="SimpleVLA-RL Logo"/>

## SimpleVLA-RL: Supercharging Vision-Language-Action (VLA) Models with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674)
[![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL)
[![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804)
[![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

> **SimpleVLA-RL unlocks efficient Reinforcement Learning for Vision-Language-Action (VLA) models, achieving state-of-the-art results with simple 0/1 rewards.**

SimpleVLA-RL provides an innovative RL framework to enhance long-horizon planning in VLA models, excelling in data-scarce environments and demonstrating superior performance in simulations and real-world tasks. Explore the code on [GitHub](https://github.com/PRIME-RL/SimpleVLA-RL).

<div align="center">
<img src="figs/teaser.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

## Key Features

*   **Enhanced Performance:** Achieves state-of-the-art results on the LIBERO benchmark.
*   **Data Efficiency:** Outperforms Supervised Fine-Tuning (SFT) with limited training data.
*   **Generalization:** Improves spatial, object, and goal generalization capabilities.
*   **"Pushcut" Phenomenon:** Reveals a novel action phenomenon improving model understanding.
*   **OpenVLA-OFT Integration:** Specifically optimized for OpenVLA-OFT models.

## News

*   **[2025-09-12]** Paper Release: The SimpleVLA-RL paper is now available: [Paper](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** Code Release: SimpleVLA-RL code is now available.

## Overview

SimpleVLA-RL introduces an effective online Reinforcement Learning (RL) approach for Vision-Language-Action (VLA) models. It utilizes simple, outcome-level 0/1 rule-based reward signals directly from simulation environments, leading to substantial performance gains.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

## Main Results

SimpleVLA-RL significantly boosts OpenVLA-OFT performance, achieving **97.6 points** on LIBERO-Long. Using only one trajectory for cold-start SFT, it elevates OpenVLA-OFT performance from 17.3 to 91.7, representing a **74.4-point improvement (430.1%)**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

## Getting Started

This section guides you through setting up and using SimpleVLA-RL.

### 1. Environment Setup

SimpleVLA-RL builds upon [veRL](https://verl.readthedocs.io/en/latest/start/install.html). Follow the instructions below to install veRL and OpenVLA-OFT:

*   **Install veRL:** Follow the official veRL installation guide: [veRL Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html).
*   **Install OpenVLA-OFT:**  Follow the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) repository.

### 2. Prepare SFT Models

You'll need a Supervised Fine-Tuning (SFT) VLA model. Options include:

*   **OpenVLA-OFT SFT Models:** Download from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Available models:
    *   `libero-10 traj1 SFT`
    *   `libero-10 trajall SFT`
*   **OpenVLA SFT Models:** Download from [Hugging Face OpenVLA models](https://huggingface.co/openvla).
*   **Other Models:** Fine-tune your own models if necessary.

### 3. Training with SimpleVLA-RL

Before running the training script, configure the following:

*   **WandB API Key:**  Update the `WANDB_API_KEY` field in `SimpleVLA-RL/align.json`.
*   **Key Variables in `examples/run_openvla_oft_rl.sh`:**
    *   `WANDB_API_KEY`: Your WandB API key.
    *   `EXPERIMENT_NAME`: Your experiment's name.
    *   `SFT_MODEL_PATH`: Path to your SFT model.
    *   `CKPT_PATH`: Save checkpoint path.
    *   `DATASET_NAME`: `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`.
    *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
    *   `NUM_GPUS`: GPUs per node (e.g., `8`).
    *   `NUM_NODES`: Nodes for RL training (e.g., `1`).

> **Note:** The script has been tested with:
>
> *   Single-node: `NUM_NODES=1`, `NUM_GPUS=8` (1 node with 8 NVIDIA A800 GPUs, 80GB memory each).
> *   Multi-node: `NUM_NODES=2`, `NUM_GPUS=8` (2 nodes with 16 NVIDIA A800 GPUs, 80GB memory each).
> *   Driver version used is `470.161.03`, and CUDA version is `12.4`. *(Not necessary)*

*   **Run RL Training:** Execute the training script:

    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

### 4. Evaluation

To evaluate your model, set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and then run the script:

```bash
bash examples/run_openvla_oft_rl.sh
```

## Acknowledgement

We are grateful to [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME) for their contributions.  Refer to their official documentation and repositories for further details.

## Contact

*   Haozhan Li: zhan72426@gmail.com
*   Ning Ding: dingning@mail.tsinghua.edu.cn

## TODO

*   **Models:**
    *   ✅ Support OpenVLA and OpenVLA-OFT
    *   ⏳ Support Pi0 fast tokenizer
*   **Benchmarks:**
    *   ✅ Support LIBERO benchmark
    *   ⏳ Support RoboTwin benchmark

## Citation

If you find SimpleVLA-RL useful, please cite our paper:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```