# RoboTwin: A Powerful Bimanual Robotic Manipulation Platform

**RoboTwin** is a cutting-edge platform designed for research and development in bimanual robotic manipulation, offering a scalable data generator, a comprehensive benchmark, and robust domain randomization capabilities. [Explore the RoboTwin Repository](https://github.com/RoboTwin-Platform/RoboTwin)

## Key Features

*   **Scalable Data Generation:** Easily generate large datasets for training and evaluating robotic manipulation algorithms.
*   **Comprehensive Benchmark:** Evaluate your algorithms on a diverse set of tasks, including those from the RoboTwin Dual-Arm Collaboration Challenge.
*   **Robust Domain Randomization:** Enhance the generalizability of your models through advanced domain randomization techniques.
*   **Multiple Versions:** Supports multiple versions (1.0, 2.0) and challenge branches for research and competition.
*   **Pre-collected Datasets:** Access over 100,000 pre-collected trajectories to jumpstart your research.
*   **Policy Baselines:** Built-in support for DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA and LLaVA-VLA baselines to get you started.
*   **Active Community:** Join the community to share ideas, collaborate, and stay up-to-date with the latest developments.

## Latest Updates

*   **2025/08/28**: RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088) updated.
*   **2025/08/25**: ACT deployment code fixed and [leaderboard](https://robotwin-platform.github.io/leaderboard) updated.
*   **2025/08/06**: RoboTwin 2.0 Leaderboard released: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **2025/07/23**: RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025.
*   **2025/07/19**: Fix DP3 evaluation code error.
*   **2025/07/09**: Endpose control mode updated.
*   **2025/07/08**: [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch released.
*   **2025/07/02**: Fix Piper Wrist Bug [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)].
*   **2025/07/01**: Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21**: RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] released.
*   **2025/04/11**: RoboTwin is selected as *CVPR Highlight paper*!
*   **2025/02/27**: RoboTwin is accepted to *CVPR 2025*!
*   **2024/09/30**: RoboTwin (Early Version) received *the Best Paper Award at the ECCV Workshop*!
*   **2024/09/20**: Officially released RoboTwin.

## Branches

*   **2.0 Version:** [main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (latest)
*   **1.0 Version:** [1.0 Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0)
*   **1.0 Version Code Generation:** [1.0 Version GPT](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt)
*   **Early Version:** [Early Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version)
*   **第十九届“挑战杯”人工智能专项赛:** [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025)
*   **CVPR 2025 Challenge Round 1:** [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1)
*   **CVPR 2025 Challenge Round 2:** [CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2)

## Installation

Follow the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for a quick and easy 20-minute installation.

## Tasks

Explore a wide variety of manipulation tasks using the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document

Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for detailed guidance.

### Data Collection

Collect your own data for tasks and embodiment setups by:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Modify Task Config

Customize tasks using configurations as described in the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   OpenVLA-oft
*   TinyVLA
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))

Deploy Your Policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

*   **Focus areas:**
    1.  Single-task fine-tuning
    2.  Visual robustness
    3.  Language diversity robustness (language condition)
    4.  Multi-task capability
    5.  Cross-embodiment performance

Find the complete leaderboard at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Access the [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

```bibtex
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

```bibtex
@InProceedings{Mu_2025_CVPR,
    author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
    title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27649-27660}
}
```

```bibtex
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

```bibtex
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgement

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos.

Contact [Tianxing Chen](https://tianxingchen.github.io) for questions or suggestions.

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.