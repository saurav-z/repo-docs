<div align="center">

<img src="figs/logo.png" width="260"/>

## 🚀 SimpleVLA-RL: Supercharging Vision-Language-Action (VLA) Models with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674) [![Github](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86) [![Twitter](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

SimpleVLA-RL is a novel reinforcement learning framework that dramatically improves VLA model performance, even with limited data, by leveraging simple 0/1 rewards. **[Check out the original repository for more details](https://github.com/PRIME-RL/SimpleVLA-RL)!**

---

## Key Features

*   **Enhanced Performance:** Achieve state-of-the-art results on the LIBERO benchmark for OpenVLA-OFT models.
*   **Data Efficiency:** Significantly boosts performance with minimal training data, showcasing strong generalization capabilities.
*   **Simple Reward Signals:** Leverages easy-to-obtain 0/1 reward signals for effective online RL.
*   **"Pushcut" Action Phenomenon:**  Reveals a new action phenomenon that leads to improvements in VLA model performance.
*   **Improved Generalization:** Strengthens spatial, object, and goal generalization within VLA models.

---

## News

*   **[2025-09-12]** The SimpleVLA-RL paper is released! [Paper](https://arxiv.org/abs/2509.09674).
*   **[2025-05-27]** Code for SimpleVLA-RL is now available.

---

## Overview

SimpleVLA-RL is an innovative RL framework for Vision-Language-Action (VLA) models, designed to improve long-horizon planning in data-scarce environments. It uses simple, outcome-level 0/1 reward signals directly from simulation environments to train models.

<div align="center">
<img src="figs/simplevla-rl.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

---

## Main Results

SimpleVLA-RL achieves impressive results on the LIBERO benchmark using OpenVLA-OFT, reaching **97.6 points**.  Remarkably, using only one trajectory per task for cold-start SFT, SimpleVLA-RL improved the performance of OpenVLA-OFT from 17.3 to 91.7, yielding an improvement of **74.4 points (430.1%)**.

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

---

## Getting Started

Follow these steps to get started with SimpleVLA-RL:

#### 1. Environment Setup

*   Install the veRL environment (see the official veRL installation guide [here](https://verl.readthedocs.io/en/latest/start/install.html)).
*   Set up OpenVLA-OFT according to the instructions in the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) repository.

#### 2. Prepare Your SFT Model

You'll need an SFT (Supervised Fine-Tuning) VLA model:

*   **OpenVLA-OFT SFT Models:** Download from the [SimpleVLA-RL Collection](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86).
    *   `libero-10 traj1 SFT`
    *   `libero-10 trajall SFT`
*   **OpenVLA SFT Models:** Download from [here](https://huggingface.co/openvla).
*   **Other Models:**  You may need to fine-tune other models yourself.

#### 3. Train with SimpleVLA-RL

Configure your training setup by setting the following parameters in the `examples/run_openvla_oft_rl.sh` file:

*   `WANDB_API_KEY`: Your WandB API key.
*   `EXPERIMENT_NAME`: Name of your experiment.
*   `SFT_MODEL_PATH`: Path to your SFT model.
*   `CKPT_PATH`: Where to save your checkpoints.
*   `DATASET_NAME`: Options include `libero_10`, `libero_90`, `libero_spatial`, `libero_object`, or `libero_goal`.
*   `ALIGN_PATH`: Path to the `SimpleVLA-RL/align.json` file.
*   `NUM_GPUS`: Number of GPUs per node.
*   `NUM_NODES`: Number of nodes used for RL training.

> [!NOTE]
> - The script has been tested on the following configurations:  
>   - Single-node setup: `NUM_NODES=1`, `NUM_GPUS=8` (1 node with 8 NVIDIA A800 GPUs, each having 80GB memory).  
>   - Multi-node setup: `NUM_NODES=2`, `NUM_GPUS=8` (2 nodes with 16 NVIDIA A800 GPUs, each having 80GB memory).  
> - The driver version used is `470.161.03`, and the CUDA version is `12.4`. *(Not necessary)*

Run training using the following command:

```bash
bash examples/run_openvla_oft_rl.sh
```

#### 4. Run Evaluation

To evaluate your model, set `trainer.val_only=True` in `examples/run_openvla_oft_rl.sh` and rerun the script:

```bash
bash examples/run_openvla_oft_rl.sh
```

---

## Acknowledgement

SimpleVLA-RL builds upon the work of [veRL](https://github.com/volcengine/verl), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [PRIME](https://github.com/PRIME-RL/PRIME).  We are grateful for their contributions.

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

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```