# RoboTwin: A Cutting-Edge Platform for Bimanual Robotic Manipulation

RoboTwin is a state-of-the-art platform for research and development in bimanual robotic manipulation, offering a scalable data generator, benchmark, and a suite of baselines. Explore the RoboTwin platform and accelerate your research at [RoboTwin's GitHub Repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate large-scale, diverse datasets for training and evaluating robotic manipulation algorithms.
*   **Robust Domain Randomization:** Implement strong domain randomization techniques to enhance the generalization capabilities of your models.
*   **Comprehensive Benchmark:** Evaluate and compare your algorithms on a challenging benchmark with a variety of bimanual manipulation tasks.
*   **Pre-collected Large-scale Dataset:** Ready-to-use dataset available on Hugging Face.
*   **Multiple Policy Baselines:** Includes support for various state-of-the-art policies, including DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Active Community:** Join a growing community of researchers and developers to share ideas and collaborate.
*   **Latest Version:** RoboTwin 2.0 with significant updates.
*   **CVPR 2025 Highlight Paper:** RoboTwin is the highlight paper at CVPR 2025.

## Key Updates

*   **2025/08/28**, We update the RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088).
*   **2025/08/25**, We fix ACT deployment code and update the [leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **2025/08/06**, We release RoboTwin 2.0 Leaderboard: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **2025/07/23**, RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/19**, We Fix DP3 evaluation code error. We will update RoboTwin 2.0 paper next week.
*   **2025/07/09**, We update endpose control mode, please see [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)] for more details.
*   **2025/07/08**, We upload [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch (第十九届挑战杯分支).
*   **2025/07/02**, Fix Piper Wrist Bug [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)]. Please redownload the embodiment asset.
*   **2025/07/01**, We release Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21**, We release RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **2025/04/11**, RoboTwin is seclected as <i>CVPR Highlight paper</i>!
*   **2025/02/27**, RoboTwin is accepted to <i>CVPR 2025</i> ! 
*   **2024/09/30**, RoboTwin (Early Version) received <i>the Best Paper Award  at the ECCV Workshop</i>!
*   **2024/09/20**, Officially released RoboTwin.

## Overview

The repository includes the following branches:

*   **2.0 Version Branch:** [main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (latest)
*   **1.0 Version Branch:** [1.0 Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0)
*   **1.0 Version Code Generation Branch:** [1.0 Version GPT](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt)
*   **Early Version Branch:** [Early Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version)
*   **第十九届“挑战杯”人工智能专项赛分支:** [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025)
*   **CVPR 2025 Challenge Round 1 Branch:** [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1)
*   **CVPR 2025 Challenge Round 2 Branch:** [CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2)

## Installation

See [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions. Installation takes about 20 minutes.

## Tasks Information

See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for task details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document

Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for comprehensive documentation.

### Data Collection

We provide over 100,000 pre-collected trajectories in the [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
However, we recommend collecting your own data to leverage the configurability and diversity of tasks and setups.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

Use the following command to collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Modify Task Config

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for configuration details.

## Policy Baselines

### Policies Supported

*   DP ([Documentation](https://robotwin-platform.github.io/doc/usage/DP.html))
*   ACT ([Documentation](https://robotwin-platform.github.io/doc/usage/ACT.html))
*   DP3 ([Documentation](https://robotwin-platform.github.io/doc/usage/DP3.html))
*   RDT ([Documentation](https://robotwin-platform.github.io/doc/usage/RDT.html))
*   PI0 ([Documentation](https://robotwin-platform.github.io/doc/usage/Pi0.html))
*   OpenVLA-oft ([Documentation](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html))
*   TinyVLA ([Documentation](https://robotwin-platform.github.io/doc/usage/TinyVLA.html))
*   DexVLA ([Documentation](https://robotwin-platform.github.io/doc/usage/DexVLA.html)) (Contributed by Media Group)
*   LLaVA-VLA ([Documentation](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html)) (Contributed by IRPN Lab, HKUST(GZ))

### Deploy Your Policy

Refer to the [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html) to deploy your own policy.

## Experiment & Leaderboard

Explore the RoboTwin Platform for:

1.  Single-task fine-tuning capability.
2.  Visual robustness.
3.  Language diversity robustness (language condition).
4.  Multi-task capabilities.
5.  Cross-embodiment performance.

View the full leaderboard: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Find the RoboTwin 2.0 Dataset on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

Please cite the following papers if you use RoboTwin:

**RoboTwin 2.0:** A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation

```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin:** Dual-Arm Robot Benchmark with Generative Digital Twins, accepted to CVPR 2025 (Highlight)

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

Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop

```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin:** Dual-Arm Robot Benchmark with Generative Digital Twins (early version), accepted to ECCV Workshop 2024 (Best Paper Award)

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

For questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT license. See [LICENSE](./LICENSE) for details.