<div align="center">

<img src="figs/logo.png" width="260" alt="SimpleVLA-RL Logo"/>

## 🚀 SimpleVLA-RL: Revolutionizing Vision-Language-Action (VLA) Model Training with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674) [![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

**SimpleVLA-RL leverages simple 0/1 rewards to unlock effective online Reinforcement Learning (RL) for Vision-Language-Action (VLA) models, achieving state-of-the-art results with data efficiency.**  This repository provides the code and resources for training VLA models using a novel RL approach. For more details, see our paper [here](https://arxiv.org/abs/2509.09674) and explore the original repository: [https://github.com/PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL).

<div align="center">
<img src="figs/teaser.png" alt="Overview of SimpleVLA-RL" width="90%" />

Overview of **SimpleVLA-RL**. SimpleVLA-RL is an efficient RL framework for VLA that improves long-horizon planning under data scarcity, outperforms SFT in simulation and real-world tasks, reveals a “pushcut” new-action phenomenon, and strengthens spatial/object/goal generalization.
</div>

## Key Features

*   **Efficient RL Framework:** Enables effective online RL for VLA models using simple 0/1 rewards.
*   **Improved Data Efficiency:**  Outperforms Supervised Fine-Tuning (SFT) with significantly fewer training examples.
*   **State-of-the-Art Performance:** Achieves new state-of-the-art results on the LIBERO benchmark.
*   **Enhanced Generalization:** Improves spatial, object, and goal generalization capabilities.
*   **"Pushcut" Phenomenon:** Reveals a novel action phenomenon in VLA models.

## 🎉 News

*   **[2025-09-12]** Paper Release: Check out the SimpleVLA-RL paper on arXiv: [https://arxiv.org/abs/2509.09674](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** Code Release:  The code for SimpleVLA-RL is now available.

## 📖 Overview

SimpleVLA-RL presents a streamlined approach to online Reinforcement Learning (RL) for Vision-Language-Action (VLA) models. It uses only outcome-level 0/1 rule-based reward signals directly from simulation environments, offering a simple yet powerful method to train VLA models.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL" width="90%" />
</div>

## 📃 Main Results

SimpleVLA-RL significantly boosts the performance of OpenVLA-OFT on the LIBERO benchmark. It achieves a score of **97.6 points** on LIBERO-Long, setting a new standard. Notably, starting from a cold-start SFT with just one trajectory per task, SimpleVLA-RL elevates OpenVLA-OFT performance from 17.3 to 91.7, a remarkable **74.4-point (430.1%) improvement**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL" width="90%" />
</div>

## ✨ Getting Started

### 1. Environment Setup

SimpleVLA-RL builds upon [veRL](https://verl.readthedocs.io/en/latest/start/install.html). Follow these steps to set up the environment, including the Vision-Language-Action (VLA) model:

*   **Install veRL:** Follow the official veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html).
*   **Install OpenVLA-OFT:**  Set up OpenVLA-OFT according to the instructions in the [OpenVLA-OFT repository](https://github.com/moojink/openvla-oft).

### 2. Prepare the SFT Model

You will need a Supervised Fine-Tuning (SFT) VLA model. Options include:

*   **OpenVLA-OFT SFT Models:** Download from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Available models:  
    *   `libero-10 traj1 SFT`
    *   `libero-10 trajall SFT`
*   **OpenVLA SFT Models:**  Download from [here](https://huggingface.co/openvla).
*   **Other Models:** You may need to fine-tune these yourself.

### 3. Training with SimpleVLA-RL

Before running the training script:

*   **Set WandB API Key:** Update the `WANDB_API_KEY` in `SimpleVLA-RL/align.json` with your API key.
*   **Modify Key Variables:** Adjust these variables in `examples/run_openvla_oft_rl.sh`:
    *   `WANDB_API_KEY`: Your WandB API key.
    *   `EXPERIMENT_NAME`: The experiment name.
    *   `SFT_MODEL_PATH`: Path to your SFT model.
    *   `CKPT_PATH`: Checkpoint save path.
    *   `DATASET_NAME`: Dataset options (e.g., `libero_10`).
    *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
    *   `NUM_GPUS`: GPUs per node.
    *   `NUM_NODES`: Number of nodes.

    > [!NOTE]
    > - Tested configurations: Single-node (`NUM_NODES=1`, `NUM_GPUS=8`) and multi-node (`NUM_NODES=2`, `NUM_GPUS=8`) setups.
    > - Driver version: `470.161.03`, CUDA version: `12.4`.

*   **Run RL Training:** Use the following command:

    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

### 4. Evaluation

To evaluate, set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and run the same script.

## 🌻 Acknowledgement

SimpleVLA-RL is developed based on [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME). We appreciate their contributions!

## 📨 Contact

*   Haozhan Li: zhan72426@gmail.com
*   Ning Ding: dingning@mail.tsinghua.edu.cn

## 📝 TODO

*   **Models:**
    *   ✅ Support OpenVLA and OpenVLA-OFT
    *   ⏳ Support Pi0 fast tokenizer
*   **Benchmarks:**
    *   ✅ Support LIBERO benchmark
    *   ⏳ Support RoboTwin benchmark

## 🎈 Citation

If SimpleVLA-RL is helpful, cite us:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```