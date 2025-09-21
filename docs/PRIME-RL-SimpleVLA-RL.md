<div align="center">

<img src="figs/logo.png" width="260" alt="SimpleVLA-RL Logo"/>

## 🚀 SimpleVLA-RL: Reinforcement Learning for Scalable VLA Training

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674)
[![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL)
[![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804)
[![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

<p align="center">
    **Supercharge your Vision-Language-Action (VLA) models with SimpleVLA-RL, a cutting-edge reinforcement learning framework for efficient and effective training.**
</p>

---

## Key Features

*   **Enhanced Performance:** SimpleVLA-RL significantly improves VLA model performance, achieving state-of-the-art results on the LIBERO benchmark.
*   **Data-Efficient Training:**  Leverages 0/1 reward signals for effective online Reinforcement Learning, even with limited data.
*   **Improved Generalization:** Enhances spatial, object, and goal generalization capabilities.
*   **"Pushcut" Action Discovery:** Uncovers a novel action phenomenon, leading to improved task performance.
*   **OpenVLA-OFT Support:**  Offers out-of-the-box support for OpenVLA-OFT models, with options to fine-tune your own.

---

## News

*   **[2025-09-12]** SimpleVLA-RL paper is released!  [Read the Paper](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** The code for SimpleVLA-RL is now available.

---

## Overview

SimpleVLA-RL introduces a novel approach to online Reinforcement Learning (RL) for Vision-Language-Action (VLA) models. It utilizes simple, outcome-level 0/1 rule-based reward signals directly from simulation environments, resulting in superior performance and improved data efficiency.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

---

## Main Results

SimpleVLA-RL sets a new state-of-the-art on the LIBERO benchmark using OpenVLA-OFT, achieving **97.6 points**. Critically, using just one trajectory per task for cold-start SFT, SimpleVLA-RL dramatically boosts OpenVLA-OFT performance from 17.3 to 91.7, marking a **74.4-point improvement (430.1%)**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

---

## Getting Started

Get up and running with SimpleVLA-RL in a few steps:

1.  **Environment Setup:**
    *   Install the veRL environment. Follow the [veRL installation guide](https://verl.readthedocs.io/en/latest/start/install.html).
    *   Install the required environment for the Vision-Language-Action (VLA) model, such as OpenVLA-OFT. For OpenVLA-OFT, follow the instructions [here](https://github.com/moojink/openvla-oft).

2.  **Prepare Your SFT Model:**
    *   **OpenVLA-OFT SFT Models:** Download pre-trained models from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Available models include:
        *   `libero-10 traj1 SFT`
        *   `libero-10 trajall SFT`
    *   **OpenVLA SFT Models:** Download from [Hugging Face](https://huggingface.co/openvla).
    *   **Other Models:** You may need to fine-tune other models yourself.

3.  **Train with SimpleVLA-RL:**
    *   **WandB API Key:**  Update the `WANDB_API_KEY` field in `SimpleVLA-RL/align.json` with your WandB API key.
    *   **Configuration:**  Modify the following variables in `examples/run_openvla_oft_rl.sh` as needed:
        *   `WANDB_API_KEY`: Your WandB API key.
        *   `EXPERIMENT_NAME`: The name of your experiment.
        *   `SFT_MODEL_PATH`: Path to your SFT model.
        *   `CKPT_PATH`: Path for saving checkpoints.
        *   `DATASET_NAME`: Choose from `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`.
        *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
        *   `NUM_GPUS`:  GPUs available per node (e.g., `8`).
        *   `NUM_NODES`: Nodes used for training (e.g., `1`).

    > [!NOTE]
    > The scripts have been tested with the following configurations:
    > * Single-node: `NUM_NODES=1`, `NUM_GPUS=8` (1 node with 8 NVIDIA A800 GPUs, 80GB memory each).
    > * Multi-node: `NUM_NODES=2`, `NUM_GPUS=8` (2 nodes with 16 NVIDIA A800 GPUs, 80GB memory each).
    > * Driver version used: `470.161.03`, CUDA version `12.4` (Not Required).

    *   **Run Training:** Execute the following command:

        ```bash
        bash examples/run_openvla_oft_rl.sh
        ```

4.  **Run Evaluation:**
    *   Set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` to enable evaluation mode.
    *   Execute the same script:

        ```bash
        bash examples/run_openvla_oft_rl.sh
        ```

---

## Acknowledgement

SimpleVLA-RL is built upon the foundation of [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME). We are grateful for their contributions!

---

## Contact

*   Haozhan Li: zhan72426@gmail.com
*   Ning Ding: dingning@mail.tsinghua.edu.cn

---

## TODO

*   **Models**:
    *   ✅ Support OpenVLA and OpenVLA-OFT
    *   ⏳ Support Pi0 fast tokenizer
*   **Benchmarks**:
    *   ✅ Support LIBERO benchmark
    *   ⏳ Support RoboTwin benchmark

---

## Citation

If you use SimpleVLA-RL, please cite us:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```

[Back to the top](https://github.com/PRIME-RL/SimpleVLA-RL)