<div align="center">

<img src="figs/logo.png" width="260"/>

## SimpleVLA-RL: Revolutionizing Vision-Language-Action (VLA) Training with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674) [![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

> **SimpleVLA-RL leverages simple 0/1 rewards to unlock effective online reinforcement learning for VLA models, achieving state-of-the-art performance.**

SimpleVLA-RL offers a powerful approach to training Vision-Language-Action (VLA) models efficiently using Reinforcement Learning (RL). This framework significantly improves long-horizon planning, especially in data-scarce scenarios.

<div align="center">
<img src="figs/teaser.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

**Key Features:**

*   **Enhanced Performance:** Achieve significant performance gains in VLA tasks.
*   **Data Efficiency:** Outperforms Supervised Fine-Tuning (SFT) with limited data.
*   **Pushcut Action Discovery:** Reveals new action behaviors in the models.
*   **Improved Generalization:** Strengthens spatial, object, and goal generalization capabilities.
*   **State-of-the-Art Results:** Achieves new state-of-the-art results on the LIBERO benchmark.

**Explore the SimpleVLA-RL repository on [GitHub](https://github.com/PRIME-RL/SimpleVLA-RL)!**

---

## 🚀 What's New

*   **[2025-09-12]** Paper release: Explore the SimpleVLA-RL paper. [Paper](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** Code release: The SimpleVLA-RL code is now available.

---

## 📖 Overview

SimpleVLA-RL introduces an efficient RL framework for Vision-Language-Action (VLA) models using online RL with simple outcome-level (0/1) rewards derived from simulation environments.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

---

## 📃 Main Results

SimpleVLA-RL significantly boosts the performance of OpenVLA-OFT on the LIBERO benchmark. Using only one trajectory per task for cold-start SFT, SimpleVLA-RL raises the performance of OpenVLA-OFT from 17.3 to 91.7, yielding an improvement of **74.4 points (430.1%)**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

---

## ✨ Getting Started

Follow these steps to get started with SimpleVLA-RL:

#### 1. Set Up the Environment

Our project builds upon [veRL](https://verl.readthedocs.io/en/latest/start/install.html). Install the veRL environment along with the environment for the Vision-Language-Action (VLA) model. Below are the detailed steps to set up **OpenVLA-OFT**.

*   **Install veRL**
    *   Follow the official veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html).

*   **Install OpenVLA-OFT**
    *   Set up OpenVLA-OFT by following the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft).

#### 2. Prepare the SFT Model

An **SFT (Supervised Fine-Tuning)** VLA model is required for RL training. Available options:

*   **OpenVLA-OFT SFT Models**
    *   Download from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Models include:
        *   `libero-10 traj1 SFT`
        *   `libero-10 trajall SFT`

*   **OpenVLA SFT Models**
    *   Download from [here](https://huggingface.co/openvla).

*   **Other Models**
    *   Fine-tune them yourself.

#### 3. Train with SimpleVLA-RL

1.  **Set Up WandB:**
    *   Replace the `WANDB_API_KEY` field in `SimpleVLA-RL/align.json` with your WandB API key.
2.  **Configure Variables:**
    *   Update these variables in `examples/run_openvla_oft_rl.sh`:
        *   `WANDB_API_KEY`: Your WandB API key.
        *   `EXPERIMENT_NAME`: Experiment name.
        *   `SFT_MODEL_PATH`: Path to your SFT model.
        *   `CKPT_PATH`: Checkpoint save path.
        *   `DATASET_NAME`: Options: `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`.
        *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
        *   `NUM_GPUS`: GPUs per node (e.g., `8`).
        *   `NUM_NODES`: Nodes used for training (e.g., `1`).
3.  **Run RL Training:**
    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

#### 4. Run Evaluation

1.  **Enable Evaluation Mode:**
    *   Set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh`.
2.  **Execute:**
    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

---

## 🌻 Acknowledgement

We extend our gratitude to the developers of [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME) for their contributions.

---

## 📨 Contact

*   Haozhan Li: zhan72426@gmail.com
*   Ning Ding: dingning@mail.tsinghua.edu.cn

---

## 📝 TODO

*   **Models:**
    *   ✅ Support OpenVLA and OpenVLA-OFT
    *   ⏳ Support Pi0 fast tokenizer
*   **Benchmarks:**
    *   ✅ Support LIBERO benchmark
    *   ⏳ Support RoboTwin benchmark

---

## 🎈 Citation

If you find SimpleVLA-RL helpful, please cite us:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```