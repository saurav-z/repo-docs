<div align="center">

<img src="figs/logo.png" width="260"/>

## SimpleVLA-RL: Revolutionizing VLA Training with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674) [![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

**SimpleVLA-RL unlocks effective online Reinforcement Learning for Vision-Language-Action (VLA) models, leveraging only 0/1 reward signals to significantly boost performance.**

[See the original repository for more details](https://github.com/PRIME-RL/SimpleVLA-RL)

## Key Features

*   **Simplified Training**: Leverages simple 0/1 rewards from simulation for effective RL.
*   **State-of-the-Art Results**: Achieves a new state-of-the-art score of 97.6 points on LIBERO-Long.
*   **Significant Performance Boost**: Improves OpenVLA-OFT performance by up to 430.1% with limited data.
*   **Enhanced Generalization**: Improves spatial, object, and goal generalization capabilities.
*   **"Pushcut" Action Phenomenon**: Reveals a novel action phenomenon observed during training.

## News

*   **[2025-09-12]** Paper Release: SimpleVLA-RL paper is now available on [arXiv](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** Code Release: SimpleVLA-RL code is now available for use.

## Overview

SimpleVLA-RL is a novel approach to online Reinforcement Learning (RL) for Vision-Language-Action (VLA) models. It utilizes outcome-level 0/1 reward signals, directly obtained from simulation environments, to enable efficient training and achieve state-of-the-art performance.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

## Main Results

SimpleVLA-RL demonstrates significant performance improvements on the LIBERO benchmark using OpenVLA-OFT. It achieves a new state-of-the-art score of **97.6 points** on LIBERO-Long. Moreover, with a cold-start SFT using only one trajectory per task, SimpleVLA-RL boosts the performance of OpenVLA-OFT from 17.3 to 91.7, a remarkable **74.4-point (430.1%) improvement**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

## Getting Started

### 1. Environment Setup

SimpleVLA-RL builds upon [veRL](https://verl.readthedocs.io/en/latest/start/install.html). Follow these steps to set up the environment:

*   **Install veRL**: Follow the official veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html).
*   **Install OpenVLA-OFT**: Follow the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) repository.

### 2. Prepare the SFT Model

You'll need a Supervised Fine-Tuning (SFT) VLA model. Options include:

*   **OpenVLA-OFT SFT Models**: Download pre-trained models from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Available models include:
    *   `libero-10 traj1 SFT`
    *   `libero-10 trajall SFT`
*   **OpenVLA SFT Models**: Download from [here](https://huggingface.co/openvla).
*   **Other Models**: You may need to fine-tune other models yourself.

### 3. Train with SimpleVLA-RL

Before running the training script, configure the following:

*   **WandB API Key**: Replace the `WANDB_API_KEY` in `SimpleVLA-RL/align.json` with your API key.
*   **Modify Variables in `examples/run_openvla_oft_rl.sh`**:
    *   `WANDB_API_KEY`: Your WandB API key.
    *   `EXPERIMENT_NAME`: Experiment name.
    *   `SFT_MODEL_PATH`: Path to your SFT model.
    *   `CKPT_PATH`: Checkpoint save path.
    *   `DATASET_NAME`: Dataset (e.g., `libero_10`).
    *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
    *   `NUM_GPUS`: GPUs per node (e.g., `8`).
    *   `NUM_NODES`: Number of nodes (e.g., `1`).

> [!NOTE]
> The script has been tested on single and multi-node setups. Driver version `470.161.03` and CUDA version `12.4` were used (not essential).

*   **Run RL Training**: Execute the following command:
    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

### 4. Run Evaluation

To evaluate your model, set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and rerun the script:

```bash
bash examples/run_openvla_oft_rl.sh
```

## Acknowledgement

We acknowledge the contributions of [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME).

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

If you find SimpleVLA-RL useful, please cite us.
```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```