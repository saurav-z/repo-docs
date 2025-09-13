# RoboTwin: A Scalable Benchmark for Bimanual Robotic Manipulation

RoboTwin is a cutting-edge platform designed to advance the field of bimanual robotic manipulation, offering a comprehensive benchmark for evaluating and developing robust and generalizable robotic solutions.  [Visit the RoboTwin GitHub Repository](https://github.com/RoboTwin-Platform/RoboTwin) to explore its capabilities!

## Key Features

*   **Scalable Data Generation:** Generate diverse and realistic datasets for training and evaluating robotic manipulation models.
*   **Strong Domain Randomization:** Implement robust domain randomization techniques to enhance the generalization capabilities of robotic systems.
*   **Bimanual Manipulation Focus:** Specifically designed for dual-arm robotic manipulation tasks.
*   **Multiple Tasks:** Supports a wide range of manipulation tasks, providing a versatile testing ground.
*   **Policy Baselines:** Offers a variety of baseline policies to facilitate benchmarking and research.
*   **Leaderboard:**  Track and compare performance using the official RoboTwin leaderboard.
*   **Comprehensive Documentation:** Detailed documentation and usage examples to help users get started quickly.

## What's New in RoboTwin 2.0

*   **RoboTwin 2.0:**  A significant update with improvements to data generation, domain randomization, and task diversity. [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088)
    *   [RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **CVPR 2025 Highlight Paper:** The original RoboTwin paper has been accepted to CVPR 2025 as a Highlight paper!  [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **CVPR 2025 Challenge:**  Participate in the RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop! [Technical Report](https://arxiv.org/abs/2506.23351)

## Overview

*   **[main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (latest):** 2.0 Version Branch
*   **[RoboTwin-1.0](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0):** 1.0 Version Branch
*   **[gpt](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt):** 1.0 Version Code Generation Branch
*   **[early_version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version):** Early Version Branch
*   **[Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025):** 第十九届“挑战杯”人工智能专项赛分支
*   **[CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1):** CVPR 2025 Challenge Round 1 Branch
*   **[CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2):** CVPR 2025 Challenge Round 2 Branch

## Installation

See the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions. Installation takes approximately 20 minutes.

## Tasks

Explore a variety of manipulation tasks in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document

Refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for detailed usage instructions.

### Data Collection

RoboTwin provides over 100,000 pre-collected trajectories.  A pre-collected dataset is available [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
However, users are encouraged to collect their own data.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

Run the following command, replacing the placeholders with your desired configuration:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Modify Task Config

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

## Policy Baselines

### Policies Support

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html)
*   [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
*   [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
*   [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
*   [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
*   [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)
*   [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) (Contributed by IRPN Lab, HKUST(GZ))

Deploy Your Policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

Explore the leaderboard and experiment with RoboTwin to explore the topics:

1.  Single-task fine-tuning capability
2.  Visual robustness
3.  Language diversity robustness (language condition)
4.  Multi-tasks capability
5.  Cross-embodiment performance

The full leaderboard and setting can be found at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Access the pre-collected dataset on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you use RoboTwin in your research, please cite the following papers:

```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

```
@InProceedings{Mu_2025_CVPR,
    author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
    title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27649-27660}
}
```

```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgement

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos.

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.