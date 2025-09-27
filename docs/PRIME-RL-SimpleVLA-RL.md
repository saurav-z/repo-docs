<div align="center">
  <img src="figs/logo.png" width="260" alt="SimpleVLA-RL Logo">
</div>

# SimpleVLA-RL: Scaling Vision-Language-Action (VLA) Training with Reinforcement Learning

**Unlock the power of VLA models with SimpleVLA-RL, achieving state-of-the-art performance using simple 0/1 reward signals.**

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674)
[![GitHub](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL)
[![Hugging Face Models](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804)
[![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

---

## Key Features

*   **State-of-the-Art Performance:** Achieve impressive results on the LIBERO benchmark.
*   **Data-Efficient Training:** Improve performance with limited data, enabling faster and more efficient training.
*   **Enhanced Generalization:** Strengthen spatial, object, and goal generalization capabilities.
*   **"Pushcut" Action Phenomenon:** Discover a new action phenomenon.
*   **Open Source & Accessible:**  Easy-to-use framework built upon existing libraries.

## Overview

SimpleVLA-RL is a novel Reinforcement Learning (RL) framework for Vision-Language-Action (VLA) models.  It leverages simple 0/1 rewards to improve long-horizon planning, outperform SFT in simulation and real-world tasks.

<div align="center">
  <img src="figs/teaser.png" alt="Overview of SimpleVLA-RL" width="90%">
  <p><em>Overview of SimpleVLA-RL.</em></p>
</div>

## Main Results

SimpleVLA-RL significantly boosts the performance of OpenVLA-OFT on the LIBERO benchmark. Using only one trajectory per task for cold-start SFT, SimpleVLA-RL dramatically improves performance, demonstrating the effectiveness of our approach.

<div align="center">
  <img src="figs/main.png" alt="Main Results of SimpleVLA-RL" width="90%">
</div>

## Getting Started

This section guides you through setting up the environment, preparing your models, and running RL training.

### 1. Environment Setup

SimpleVLA-RL builds upon [veRL](https://verl.readthedocs.io/en/latest/start/install.html). Follow the instructions below to install veRL and set up the environment, specifically for the Vision-Language-Action (VLA) model.

*   **Install veRL:** Follow the official veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html).
*   **Install OpenVLA-OFT:**  Set up OpenVLA-OFT by following the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) repository.

### 2. Prepare the SFT Model

An SFT (Supervised Fine-Tuning) VLA model is essential for RL training. You can choose from the following options:

*   **OpenVLA-OFT SFT Models:** Download pre-trained models from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Available models include:
    *   `libero-10 traj1 SFT`
    *   `libero-10 trajall SFT`
*   **OpenVLA SFT Models:** Download models from [here](https://huggingface.co/openvla).
*   **Other Models:** You may need to fine-tune other models yourself.

### 3. Training with SimpleVLA-RL

Before training, configure the following settings:

*   **WandB API Key:**  Replace the `WANDB_API_KEY` field in `SimpleVLA-RL/align.json` with your API key.
*   **Key Variable Modification:** Update the variables in `examples/run_openvla_oft_rl.sh`:
    *   `WANDB_API_KEY`: Your WandB API key.
    *   `EXPERIMENT_NAME`: Experiment name.
    *   `SFT_MODEL_PATH`: Path to your SFT model.
    *   `CKPT_PATH`: Checkpoint save path.
    *   `DATASET_NAME`:  Dataset selection (e.g., `libero_10`).
    *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
    *   `NUM_GPUS`: GPUs per node (e.g., `8`).
    *   `NUM_NODES`: Number of nodes (e.g., `1`).

> [!NOTE]
> The script has been tested on single and multi-node setups (see original README for details).

*   **Run RL Training:** Execute the following command:

    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

### 4. Run Evaluation

To evaluate your model, set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and run the script again.

```bash
bash examples/run_openvla_oft_rl.sh
```

## Acknowledgment

We acknowledge the contributions from  [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME), which served as the foundation of this project.  See their respective repositories for more details.

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

---

**[Visit the SimpleVLA-RL GitHub Repository](https://github.com/PRIME-RL/SimpleVLA-RL) to get started!**