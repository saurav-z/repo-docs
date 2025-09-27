# RoboTwin: The Advanced Bimanual Robotic Manipulation Platform

RoboTwin is a cutting-edge platform designed for research and development in bimanual robotic manipulation, offering a scalable data generator, benchmark, and a strong domain randomization environment for robust performance.  [Explore the original RoboTwin repository on GitHub](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Scalable Data Generation:** Generate large-scale, diverse datasets for training and evaluating bimanual robotic manipulation models.
*   **Strong Domain Randomization:**  Utilize robust domain randomization techniques to enhance the generalization capabilities of learned policies.
*   **Comprehensive Benchmark:** Provides a standardized benchmark with various tasks to evaluate and compare different algorithms.
*   **Multiple Policy Baselines:** Includes support for various state-of-the-art policies, including DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Pre-collected Dataset:** Access a pre-collected, large-scale dataset ([RoboTwin 2.0 Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset)) to accelerate research and development.
*   **Active Community:** Join the RoboTwin community to connect with other researchers, share knowledge, and contribute to the platform.

## Latest Updates

*   **2025/08/28**: RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088) updated.
*   **2025/08/25**: ACT deployment code fixed and [leaderboard](https://robotwin-platform.github.io/leaderboard) updated.
*   **2025/08/06**: RoboTwin 2.0 Leaderboard released: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **2025/07/23**: RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/19**: DP3 evaluation code error fixed. RoboTwin 2.0 paper update coming next week.
*   **2025/07/09**: Endpose control mode updated, details in [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)].
*   **2025/07/08**: [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch (第十九届挑战杯分支) uploaded.
*   **2025/07/02**: Piper Wrist Bug fixed [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)]. Redownload the embodiment asset.
*   **2025/07/01**: Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop released [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21**: RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] released!
*   **2025/04/11**: RoboTwin selected as *CVPR Highlight paper*!
*   **2025/02/27**: RoboTwin accepted to *CVPR 2025*!
*   **2024/09/30**: RoboTwin (Early Version) received *the Best Paper Award at the ECCV Workshop*!
*   **2024/09/20**: Officially released RoboTwin.

## Branch Overview

*   **2.0 Version Branch:** [main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (latest)
*   **1.0 Version Branch:** [1.0 Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0)
*   **1.0 Version Code Generation Branch:** [1.0 Version GPT](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt)
*   **Early Version Branch:** [Early Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version)
*   **第十九届“挑战杯”人工智能专项赛分支:** [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025)
*   **CVPR 2025 Challenge Round 1 Branch:** [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1)
*   **CVPR 2025 Challenge Round 2 Branch:** [CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2)

## Installation

Detailed installation instructions can be found in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation takes approximately 20 minutes.

## Tasks Information

Explore the details of available tasks in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

![RoboTwin Tasks](https://github.com/RoboTwin-Platform/RoboTwin/raw/main/assets/files/50_tasks.gif)

## Usage

### Document

Refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for comprehensive usage details.

### Data Collection

The platform provides over 100,000 pre-collected trajectories as part of the open-source [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).  However, collecting your own data is highly encouraged for customization and diversity.

![Domain Randomization Example](https://github.com/RoboTwin-Platform/RoboTwin/raw/main/assets/files/domain_randomization.png)

### Example: Task Running and Data Collection

Run the following command to collect data by searching for a random seed:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Modify Task Config

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for configuration details.

## Policy Baselines

### Policies Supported

DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA (Contributed by Media Group), LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ)).

### Deploy Your Policy

Refer to the [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html) for guidance on deploying your own policies.

## Experiment & Leaderboard

The RoboTwin Platform enables exploration of various research areas, including single-task fine-tuning, visual robustness, language diversity robustness, multi-task capabilities, and cross-embodiment performance.

The full leaderboard is available at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Find the dataset on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you find RoboTwin useful, please cite the following:

**RoboTwin 2.0:**
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin:**
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

**Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop:**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (early version):**
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