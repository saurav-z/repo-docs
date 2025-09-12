<div align="center">
  <img src="figs/logo.png" width="260" alt="SimpleVLA-RL Logo"/>
</div>

# SimpleVLA-RL: Revolutionizing Vision-Language-Action (VLA) Learning with Reinforcement Learning

**Unlock the potential of VLA models with SimpleVLA-RL, a straightforward yet powerful reinforcement learning framework, achieving state-of-the-art performance in complex tasks.  <a href="https://github.com/PRIME-RL/SimpleVLA-RL">Explore the code!</a>**

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674)
[![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL)
[![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804)

---

## Key Features

*   **Simple and Effective RL:** Leverages outcome-level 0/1 rewards directly from simulation environments to train VLA models.
*   **State-of-the-Art Performance:** Achieves exceptional results on the LIBERO benchmark, surpassing existing methods.
*   **Data Efficiency:** Significantly improves performance with limited training data, demonstrating robust learning even from a single trajectory.
*   **Generalization:** Enhances spatial, object, and goal generalization capabilities of VLA models.
*   **Open-Source & Accessible:** Provides easily reproducible code and pre-trained models.

---

<div align="center">
<img src="figs/teaser.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

## News

*   **[2025-09-12]**  Paper release!  Check out the SimpleVLA-RL paper on [arXiv](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]**  Code is now available!

---

## Overview

SimpleVLA-RL introduces a novel approach to online Reinforcement Learning (RL) for Vision-Language-Action (VLA) models.  This framework leverages the simplicity of 0/1 rewards derived directly from simulation environments.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

---

## Main Results

SimpleVLA-RL demonstrates significant performance gains when evaluated on LIBERO using OpenVLA-OFT. It achieves **97.6 points** on LIBERO-Long, setting a new state-of-the-art.  Moreover, with just one trajectory for cold-start SFT, SimpleVLA-RL boosts OpenVLA-OFT performance from 17.3 to 91.7, an impressive **74.4-point (430.1%)** improvement.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="60%" />
</div>

---

## Getting Started

This section guides you through setting up the environment, preparing the SFT model, and running RL training and evaluation.

1.  **Set Up the Environment:**

    *   Install the veRL environment following the official [veRL installation guide](https://verl.readthedocs.io/en/latest/start/install.html).
    *   Install OpenVLA-OFT following the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) repository.

2.  **Prepare the SFT Model:**

    *   Download pre-trained SFT models from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Available models:
        *   `libero-10 traj1 SFT`
        *   `libero-10 trajall SFT`
    *   Alternatively, download models from [OpenVLA](https://huggingface.co/openvla).
    *   Fine-tune other models as needed.

3.  **Train with SimpleVLA-RL:**

    *   **Configure Settings:**

        *   Replace the `WANDB_API_KEY` field in `SimpleVLA-RL/align.json` with your WandB API key.
        *   Modify variables in `examples/run_openvla_oft_rl.sh`:
            *   `WANDB_API_KEY`: Your WandB API key.
            *   `EXPERIMENT_NAME`: Experiment name.
            *   `SFT_MODEL_PATH`: Path to your SFT model.
            *   `CKPT_PATH`: Path to save checkpoints.
            *   `DATASET_NAME`: e.g., `libero_10`, `libero_90`, etc.
            *   `ALIGN_PATH`: Path to `SimpleVLA-RL/align.json`.
            *   `NUM_GPUS`: GPUs per node (e.g., `8`).
            *   `NUM_NODES`: Number of nodes (e.g., `1`).

    *   **Run RL Training:** Execute the following command:

        ```bash
        bash examples/run_openvla_oft_rl.sh
        ```

4.  **Run Evaluation:**

    *   Set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh`.
    *   Run the same script:

        ```bash
        bash examples/run_openvla_oft_rl.sh
        ```

---

## Acknowledgement

We build upon the foundations of [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME). We are grateful for their significant contributions!

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

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```