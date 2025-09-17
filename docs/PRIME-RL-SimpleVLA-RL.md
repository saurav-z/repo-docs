<div align="center">

<img src="figs/logo.png" width="260" alt="SimpleVLA-RL Logo"/>

## SimpleVLA-RL: Revolutionizing Vision-Language-Action (VLA) Training with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674)
[![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL)
[![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804)
[![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

SimpleVLA-RL leverages simple 0/1 rewards to dramatically improve Vision-Language-Action (VLA) model training via Reinforcement Learning.  **[Explore the SimpleVLA-RL Repository](https://github.com/PRIME-RL/SimpleVLA-RL)**

---

## Key Features

*   **Effective RL with Simple Rewards:**  Achieves strong performance using only outcome-level, 0/1 rule-based reward signals.
*   **Significant Performance Gains:** Improves OpenVLA-OFT performance, achieving state-of-the-art results, including a 74.4-point (430.1%) improvement with limited data.
*   **Addresses Data Scarcity:** Excels in long-horizon planning tasks, particularly under data-scarce conditions.
*   **Enhanced Generalization:** Improves spatial, object, and goal generalization capabilities.
*   **"Pushcut" Action Discovery:**  Reveals a novel "pushcut" action phenomenon within the VLA models.

---

## News

*   **[2025-09-12]** The SimpleVLA-RL paper is released! [Read the Paper](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** Code for SimpleVLA-RL is now available.

---

## Overview

SimpleVLA-RL provides a streamlined approach to Reinforcement Learning (RL) for Vision-Language-Action (VLA) models. This approach utilizes only 0/1 reward signals derived directly from simulation environments.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

---

## Main Results

SimpleVLA-RL achieves state-of-the-art performance on the LIBERO benchmark using OpenVLA-OFT, reaching **97.6 points** on LIBERO-Long. Critically, it significantly boosts performance from a cold-start, single-trajectory SFT, improving OpenVLA-OFT's score from 17.3 to 91.7 points, a substantial gain of **74.4 points (430.1%)**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

---

## Getting Started

Follow these steps to set up and run SimpleVLA-RL.

### 1. Environment Setup

1.  **Install veRL:** Follow the veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html).
2.  **Install OpenVLA-OFT:** Set up OpenVLA-OFT by following the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft).

### 2. Prepare the SFT Model

Select and prepare a Supervised Fine-Tuning (SFT) VLA model:

*   **OpenVLA-OFT SFT Models:** Download from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Example models include:
    *   `libero-10 traj1 SFT`
    *   `libero-10 trajall SFT`
*   **OpenVLA SFT Models:** Download from [here](https://huggingface.co/openvla).
*   **Other Models:**  Fine-tune your models as needed.

### 3. Training with SimpleVLA-RL

Configure and run the training script:

1.  **WandB API Key:** Set your Weights and Biases (WandB) API key in `SimpleVLA-RL/align.json`.
2.  **Key Variable Modifications:** Update variables in `examples/run_openvla_oft_rl.sh`:
    *   `WANDB_API_KEY`: Your WandB API key.
    *   `EXPERIMENT_NAME`: Your experiment name.
    *   `SFT_MODEL_PATH`: Path to your SFT model.
    *   `CKPT_PATH`: Path for saving checkpoints.
    *   `DATASET_NAME`: Choose from `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`.
    *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
    *   `NUM_GPUS`: GPUs per node (e.g., `8`).
    *   `NUM_NODES`: Nodes used for RL training (e.g., `1`).
3.  **Run RL Training:**  Execute the script:

    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

### 4. Run Evaluation

To evaluate, set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and then run the script:

```bash
bash examples/run_openvla_oft_rl.sh
```

---

## Acknowledgement

This project builds upon the contributions of [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME).

---

## Contact

*   Haozhan Li: zhan72426@gmail.com
*   Ning Ding: dingning@mail.tsinghua.edu.cn

---

## TODO

*   **Models:**
    *   ✅ Support OpenVLA and OpenVLA-OFT
    *   ⏳ Support Pi0 fast tokenizer
*   **Benchmarks:**
    *   ✅ Support LIBERO benchmark
    *   ⏳ Support RoboTwin benchmark

---

## Citation

If you find SimpleVLA-RL helpful, please cite us:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```