<div align="center">

<img src="figs/logo.png" width="260"/>

## SimpleVLA-RL: Revolutionizing VLA Training with Reinforcement Learning

[**SimpleVLA-RL**](https://github.com/PRIME-RL/SimpleVLA-RL) introduces a novel reinforcement learning approach to efficiently train Vision-Language-Action (VLA) models, achieving state-of-the-art results with simple rewards.

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674) [![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

## Key Features

*   **Enhanced Performance:** Achieves state-of-the-art results on the LIBERO benchmark and improves long-horizon planning under data scarcity.
*   **Data Efficiency:** SimpleVLA-RL dramatically improves the performance of OpenVLA-OFT, demonstrating an improvement of 74.4 points (430.1%) using only one trajectory per task for cold-start SFT.
*   **Novel Insights:** Reveals a "pushcut" new-action phenomenon.
*   **Improved Generalization:** Strengthens spatial, object, and goal generalization capabilities in VLA models.
*   **Easy to Use:** Provides clear instructions for setup and training, allowing researchers and practitioners to quickly implement and experiment with the framework.

<div align="center">
<img src="figs/teaser.png" alt="Overview of SimpleVLA-RL." width="90%" />
Overview of **SimpleVLA-RL**. SimpleVLA-RL is an efficient RL framework for VLA that improves long-horizon planning under data scarcity, outperforms SFT in simulation and real-world tasks, reveals a “pushcut” new-action phenomenon, and strengthens spatial/object/goal generalization.
</div>

## News

*   **September 12, 2025:** [Paper](https://arxiv.org/abs/2509.09674) Release!
*   **May 27, 2025:** Code Release!

## Overview

SimpleVLA-RL is a straightforward yet powerful Reinforcement Learning (RL) approach tailored for Vision-Language-Action (VLA) models.  It leverages simple, outcome-level 0/1 rule-based reward signals directly from simulation environments to drive efficient training.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

## Main Results

SimpleVLA-RL significantly boosts OpenVLA-OFT performance.  It achieves **97.6 points** on LIBERO-Long and sets a new state-of-the-art.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

## Getting Started

Get up and running with SimpleVLA-RL in a few simple steps:

1.  **Environment Setup:**
    *   Install the veRL environment. Follow the official veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html).
    *   Install OpenVLA-OFT.  Follow instructions from [OpenVLA-OFT](https://github.com/moojink/openvla-oft).

2.  **Prepare the SFT Model:**
    *   Download pre-trained **OpenVLA-OFT SFT Models** from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86):
        *   `libero-10 traj1 SFT`
        *   `libero-10 trajall SFT`
    *   Alternatively, use **OpenVLA SFT Models** from [here](https://huggingface.co/openvla).
    *   For other models, you'll need to fine-tune them yourself.

3.  **Training with SimpleVLA-RL:**

    *   **WandB Configuration:** Set your Weights and Biases (WandB) API key in `SimpleVLA-RL/align.json`.

    *   **Script Customization:** Modify the following variables in `examples/run_openvla_oft_rl.sh` as needed:

        *   `WANDB_API_KEY`: Your WandB API key.
        *   `EXPERIMENT_NAME`: Your experiment name.
        *   `SFT_MODEL_PATH`: Path to your SFT model.
        *   `CKPT_PATH`: Path to save checkpoints.
        *   `DATASET_NAME`: Select from `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`.
        *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
        *   `NUM_GPUS`: GPUs per node (e.g., `8`).
        *   `NUM_NODES`: Number of nodes (e.g., `1`).

    >   **Note:** Tested configurations: Single-node (`NUM_NODES=1`, `NUM_GPUS=8`) and multi-node (`NUM_NODES=2`, `NUM_GPUS=8`).  Driver version `470.161.03` and CUDA `12.4` were used.

    *   **Run Training:**  Execute the following command:

        ```bash
        bash examples/run_openvla_oft_rl.sh
        ```

4.  **Evaluation:** Enable evaluation mode by setting `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and run the script.

## Acknowledgement

This work builds upon the foundations laid by [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME).  We are grateful for their contributions.  See their documentation for more details.

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

If you use SimpleVLA-RL, please cite our paper:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```