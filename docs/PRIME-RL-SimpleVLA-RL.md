<div align="center">

<img src="figs/logo.png" width="260"/>

## SimpleVLA-RL: Revolutionizing VLA Training with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674) [![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

**SimpleVLA-RL empowers Vision-Language-Action (VLA) models with efficient online reinforcement learning, achieving state-of-the-art performance with simple 0/1 rewards.**  [Explore the SimpleVLA-RL Repository](https://github.com/PRIME-RL/SimpleVLA-RL)

<div align="center">
<img src="figs/teaser.png" alt="Overview of SimpleVLA-RL." width="90%" />

Overview of **SimpleVLA-RL**. SimpleVLA-RL is an efficient RL framework for VLA that improves long-horizon planning under data scarcity, outperforms SFT in simulation and real-world tasks, reveals a “pushcut” new-action phenomenon, and strengthens spatial/object/goal generalization.
</div>

## Key Features

*   **Simplified RL Approach:** Leverages 0/1 reward signals for effective online reinforcement learning.
*   **Enhanced Performance:**  Significantly improves the performance of OpenVLA-OFT, achieving new state-of-the-art results.
*   **Data-Efficient Training:**  Demonstrates remarkable gains with limited training data, drastically improving performance from cold-start SFT.
*   **Improved Generalization:**  Strengthens spatial, object, and goal generalization capabilities in VLA models.
*   **OpenVLA-OFT Support:**  Fully supports OpenVLA-OFT models.

## News

*   **[2025-09-12]** SimpleVLA-RL paper released!  Read the full paper: [Paper](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** Code for SimpleVLA-RL is now available!

## Overview

SimpleVLA-RL presents a novel approach to online Reinforcement Learning (RL) for Vision-Language-Action (VLA) models, focusing on efficiency and performance.  It utilizes simple outcome-level, 0/1 rule-based reward signals directly from simulation environments.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

## Main Results

SimpleVLA-RL achieves impressive results when evaluated on the LIBERO benchmark using OpenVLA-OFT, setting a new state-of-the-art with a score of **97.6 points**. SimpleVLA-RL shows impressive gains using limited data: starting from a cold-start SFT model, it raises the performance of OpenVLA-OFT from 17.3 to 91.7, a **74.4-point (430.1%) improvement**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

## Getting Started

Follow these steps to set up and run SimpleVLA-RL.

### 1. Set Up the Environment

SimpleVLA-RL builds on [veRL](https://verl.readthedocs.io/en/latest/start/install.html).  You'll need to install veRL and the required environment for your chosen VLA model (e.g., OpenVLA-OFT).

*   **Install veRL:** Follow the official veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html).
*   **Install OpenVLA-OFT:**  Follow the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) repository.

### 2. Prepare the SFT Model

You'll need a Supervised Fine-Tuning (SFT) VLA model.  Options include:

*   **OpenVLA-OFT SFT Models:**  Download from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Available models include:
    *   `libero-10 traj1 SFT`
    *   `libero-10 trajall SFT`
*   **OpenVLA SFT Models:** Download from [here](https://huggingface.co/openvla).
*   **Other Models:** You may need to fine-tune other models yourself.

### 3. Train with SimpleVLA-RL

Before training, configure your environment:

*   **WandB API Key:**  Update the `WANDB_API_KEY` field in `SimpleVLA-RL/align.json` with your WandB API key.
*   **Key Variables:** Modify the following variables in `examples/run_openvla_oft_rl.sh`:
    *   `WANDB_API_KEY`: Your WandB API key.
    *   `EXPERIMENT_NAME`: Your experiment name.
    *   `SFT_MODEL_PATH`: Path to your SFT model.
    *   `CKPT_PATH`: Where checkpoints will be saved.
    *   `DATASET_NAME`: Choose from: `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`.
    *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
    *   `NUM_GPUS`: GPUs per node (e.g., `8`).
    *   `NUM_NODES`: Number of nodes (e.g., `1`).

>   **Note:** The script has been tested on configurations using NVIDIA A800 GPUs.

*   **Run RL Training:**  Execute this command to start training for OpenVLA-OFT on the LIBERO benchmark:
    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

### 4. Run Evaluation

To evaluate your model, set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and run the script:
```bash
bash examples/run_openvla_oft_rl.sh
```

## Acknowledgement

This code builds upon the work of [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME). We are grateful for their contributions.

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

If you find this project helpful, please cite our paper:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```