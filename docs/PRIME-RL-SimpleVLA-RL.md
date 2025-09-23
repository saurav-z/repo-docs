<div align="center">

<img src="figs/logo.png" width="260" alt="SimpleVLA-RL Logo"/>

## SimpleVLA-RL: Reinforcement Learning for Scalable Vision-Language-Action (VLA) Models

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674)
[![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL)
[![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804)
[![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

> **Unlock the potential of Vision-Language-Action (VLA) models with SimpleVLA-RL, a simple yet powerful reinforcement learning framework achieving state-of-the-art results with minimal data.**

**[Visit the SimpleVLA-RL GitHub Repository](https://github.com/PRIME-RL/SimpleVLA-RL)**

## Key Features

*   **Improved Performance:** SimpleVLA-RL significantly boosts the performance of OpenVLA-OFT, achieving state-of-the-art results on the LIBERO benchmark.
*   **Data Efficiency:** Dramatically improves performance with limited data; for example, increasing OpenVLA-OFT performance by 74.4 points (430.1%) using only one trajectory per task.
*   **"Pushcut" Action Phenomenon:** The framework reveals a novel action phenomenon, contributing to more efficient learning.
*   **Enhanced Generalization:** Strengthens spatial, object, and goal generalization capabilities of VLA models.
*   **Open-Source & Accessible:**  Includes detailed instructions and resources for easy implementation and experimentation.

## News

*   **[2025-09-12]** The **SimpleVLA-RL** paper is now available!  [Paper](https://arxiv.org/abs/2509.09674)
*   **[2025-05-27]** The code for **SimpleVLA-RL** is publicly released.

## Overview

SimpleVLA-RL is an efficient Reinforcement Learning (RL) framework designed for Vision-Language-Action (VLA) models. It leverages simple 0/1 reward signals from simulation environments to improve long-horizon planning, data efficiency, and generalization capabilities.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

## Main Results

SimpleVLA-RL achieves a state-of-the-art score of **97.6 points** on LIBERO-Long with OpenVLA-OFT. Furthermore, using only one trajectory per task in a cold-start SFT scenario, SimpleVLA-RL dramatically improves OpenVLA-OFT from 17.3 to 91.7 points, an impressive **74.4 point (430.1%) increase**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

## Getting Started

Follow these steps to get started with SimpleVLA-RL:

#### 1. Set Up the Environment

SimpleVLA-RL relies on [veRL](https://verl.readthedocs.io/en/latest/start/install.html) and requires the environment for your VLA model (e.g., OpenVLA-OFT).

*   **Install veRL:**  Follow the veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html).
*   **Install OpenVLA-OFT:** Follow the OpenVLA-OFT setup instructions [here](https://github.com/moojink/openvla-oft).

#### 2. Prepare the SFT Model

You'll need a Supervised Fine-Tuning (SFT) VLA model.  Available options include:

*   **OpenVLA-OFT SFT Models:** Download from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Example models: `libero-10 traj1 SFT`, `libero-10 trajall SFT`.
*   **OpenVLA SFT Models:** Download from [here](https://huggingface.co/openvla).
*   **Other Models:** Fine-tune your own model if needed.

#### 3. Train with SimpleVLA-RL

Before running the training script, configure the following:

*   **WandB API Key:** Replace the `WANDB_API_KEY` field in `SimpleVLA-RL/align.json` with your key.
*   **Key Variables:** Update these in `examples/run_openvla_oft_rl.sh`:
    *   `WANDB_API_KEY`: Your WandB API key.
    *   `EXPERIMENT_NAME`: Your experiment's name.
    *   `SFT_MODEL_PATH`: Path to your SFT model.
    *   `CKPT_PATH`: Path for checkpoint saving.
    *   `DATASET_NAME`: Dataset options: `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`.
    *   `ALIGN_PATH`: Path to the `SimpleVLA-RL/align.json` file.
    *   `NUM_GPUS`: GPUs per node (e.g., `8`).
    *   `NUM_NODES`: Number of nodes (e.g., `1`).

    > **Note:** Tested configurations include single-node (`NUM_NODES=1`, `NUM_GPUS=8`) and multi-node (`NUM_NODES=2`, `NUM_GPUS=8`) setups.  CUDA version 12.4, driver version 470.161.03

*   **Run RL Training:** Execute the following command:

    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

#### 4. Run Evaluation

To evaluate your model, set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and run the script again.

```bash
bash examples/run_openvla_oft_rl.sh
```

## Acknowledgement

We are grateful to [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME) for their significant contributions. Refer to their respective repositories for further details.

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

If you find this work helpful, please cite it:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```