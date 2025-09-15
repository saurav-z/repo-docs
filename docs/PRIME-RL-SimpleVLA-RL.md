<div align="center">

<img src="figs/logo.png" width="260"/>

## 🚀 SimpleVLA-RL: Revolutionizing VLA Training with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674)
[![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL)
[![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804)
[![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

> **SimpleVLA-RL leverages simple 0/1 rewards to achieve effective online RL for Vision-Language-Action (VLA) models, significantly enhancing their performance.**

## Key Features

*   **Enhanced Performance:** Achieve state-of-the-art results on the LIBERO benchmark and outperform SFT in simulation and real-world tasks.
*   **Data Efficiency:** Significantly improves OpenVLA-OFT performance with limited training data, showcasing strong generalization capabilities.
*   **Novel Discoveries:** Reveals a unique "pushcut" new-action phenomenon within the VLA model framework.
*   **Improved Generalization:** Strengthens spatial, object, and goal generalization.
*   **Open Source:** The code is available on GitHub: [PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)

## News

*   **[2025-09-12]** Paper Release! Explore the full details in our [SimpleVLA-RL paper](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** Code Release! The SimpleVLA-RL code is now available for use.

## Overview

SimpleVLA-RL is an innovative reinforcement learning framework for Vision-Language-Action (VLA) models.  It uses outcome-level 0/1 rule-based rewards from simulation environments for efficient online RL, resulting in significant performance gains.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

## Main Results

SimpleVLA-RL dramatically improves OpenVLA-OFT performance on the LIBERO benchmark, achieving a new state-of-the-art of **97.6 points** on LIBERO-Long. Using a cold-start SFT with only one trajectory per task, SimpleVLA-RL boosts performance from 17.3 to 91.7, an impressive **74.4-point (430.1%) improvement**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

## Getting Started

Follow these steps to get started with SimpleVLA-RL:

#### 1. Set Up the Environment

SimpleVLA-RL builds upon [veRL](https://verl.readthedocs.io/en/latest/start/install.html).  Install veRL and the necessary environment for your VLA model (OpenVLA-OFT):

*   **Install veRL:** Follow the veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html).
*   **Install OpenVLA-OFT:** Set up OpenVLA-OFT by following the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) repository.

#### 2. Prepare the SFT Model

You'll need a Supervised Fine-Tuning (SFT) VLA model for RL training. Choose from the following:

*   **OpenVLA-OFT SFT Models:** Download models from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Example models:  `libero-10 traj1 SFT`,  `libero-10 trajall SFT`.
*   **OpenVLA SFT Models:** Download from [here](https://huggingface.co/openvla).
*   **Other Models:** Fine-tune other models yourself.

#### 3. Train with SimpleVLA-RL

Before running the training script, configure these settings:

*   **WandB API Key:**  Set your WandB API key in `SimpleVLA-RL/align.json`.
*   **Key Variables:** Update these variables in `examples/run_openvla_oft_rl.sh`:
    *   `WANDB_API_KEY`: Your WandB API key.
    *   `EXPERIMENT_NAME`: Experiment name.
    *   `SFT_MODEL_PATH`: Path to your SFT model.
    *   `CKPT_PATH`: Checkpoint save path.
    *   `DATASET_NAME`: Dataset (e.g., `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`).
    *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
    *   `NUM_GPUS`: GPUs per node (e.g., `8`).
    *   `NUM_NODES`: Number of nodes (e.g., `1`).

> [!NOTE]
>   - Tested Configurations: Single-node (`NUM_NODES=1`, `NUM_GPUS=8`) and multi-node (`NUM_NODES=2`, `NUM_GPUS=8`).
>   -  Driver and CUDA versions used are `470.161.03` and `12.4` respectively.

*   **Run RL Training:** Execute this command to start training for OpenVLA-OFT:

    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

#### 4. Run Evaluation

To evaluate your model, set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and then run the script:

```bash
bash examples/run_openvla_oft_rl.sh
```

## Acknowledgement

This code is based on contributions from [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME).  We appreciate their significant contributions.

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

If you use SimpleVLA-RL, please cite our work:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```