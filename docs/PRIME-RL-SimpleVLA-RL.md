<div align="center">

<img src="figs/logo.png" width="260" alt="SimpleVLA-RL Logo"/>

## 🚀 SimpleVLA-RL: Revolutionizing Vision-Language-Action (VLA) Models with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674)
[![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL)
[![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86)
[![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804)
[![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

**SimpleVLA-RL leverages reinforcement learning with simple 0/1 rewards to effectively train VLA models, achieving state-of-the-art performance.**

<div align="center">
<img src="figs/teaser.png" alt="Overview of SimpleVLA-RL." width="90%" />

Overview of **SimpleVLA-RL**. SimpleVLA-RL is an efficient RL framework for VLA that improves long-horizon planning under data scarcity, outperforms SFT in simulation and real-world tasks, reveals a “pushcut” new-action phenomenon, and strengthens spatial/object/goal generalization.
</div>

## Key Features

*   **Improved Performance:** Achieves state-of-the-art results on the LIBERO benchmark.
*   **Data Efficiency:** Significantly boosts performance with limited training data.
*   **Novel Discoveries:** Uncovers the "pushcut" action phenomenon.
*   **Enhanced Generalization:** Strengthens spatial, object, and goal generalization capabilities.
*   **Simple Approach:** Utilizes straightforward 0/1 reward signals for effective RL training.

## News

*   **[2025-09-12]** Paper release! [Read the paper](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** Code release of **SimpleVLA-RL**.

## Overview

SimpleVLA-RL is an innovative approach to Reinforcement Learning (RL) for Vision-Language-Action (VLA) models. It simplifies the RL process by relying on outcome-level 0/1 rule-based rewards directly from simulation environments, enabling effective and efficient training.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

## Main Results

SimpleVLA-RL improves the performance of OpenVLA-OFT to **97.6 points** on LIBERO-Long, a new state-of-the-art. Notably, starting from a cold-start SFT with only one trajectory per task, SimpleVLA-RL boosts the performance of OpenVLA-OFT from 17.3 to 91.7, with an impressive improvement of **74.4 points (430.1%)**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

## Getting Started

Follow these steps to get started with SimpleVLA-RL:

#### 1.  Set Up the Environment

*   **Install veRL:** Follow the veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html).
*   **Install OpenVLA-OFT:** Set up OpenVLA-OFT by following the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft).

#### 2. Prepare the SFT Model

*   **Download Pre-trained Models:** Download from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86). Available models include:
    *   `libero-10 traj1 SFT`
    *   `libero-10 trajall SFT`
*   **Other Options:**
    *   Download OpenVLA SFT Models from [here](https://huggingface.co/openvla).
    *   Fine-tune your own model.

#### 3.  Train with SimpleVLA-RL

*   **Configure WandB:** Replace the `WANDB_API_KEY` field in `SimpleVLA-RL/align.json` with your WandB API key.
*   **Modify Configuration Variables:** Update the following variables in `examples/run_openvla_oft_rl.sh`:
    *   `WANDB_API_KEY`: Your WandB API key.
    *   `EXPERIMENT_NAME`: The name of your experiment.
    *   `SFT_MODEL_PATH`: Path to your SFT model.
    *   `CKPT_PATH`: Path to save checkpoints.
    *   `DATASET_NAME`: Options include `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`.
    *   `ALIGN_PATH`: Path to the `SimpleVLA-RL/align.json` file.
    *   `NUM_GPUS`: Number of GPUs per node (e.g., `8`).
    *   `NUM_NODES`: Number of nodes (e.g., `1`).
*   **Run RL Training:** Execute the training script:

    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

#### 4.  Run Evaluation

*   Set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` to enable evaluation mode.
*   Run the script:

    ```bash
    bash examples/run_openvla_oft_rl.sh
    ```

## Acknowledgement

This project builds upon the contributions of [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME).

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

If you find this project useful, please cite our paper:

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```

---

[Back to Top](#) (Link to the top of the document)