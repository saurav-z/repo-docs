# RoboTwin: The Ultimate Bimanual Robotic Manipulation Platform

**RoboTwin** is a cutting-edge bimanual robotic manipulation platform designed for research and development, offering a comprehensive benchmark and versatile data generation capabilities. Explore the [original RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate massive datasets with strong domain randomization for robust bimanual robotic manipulation.
*   **Comprehensive Benchmark:** Evaluate your algorithms with a standardized set of challenging tasks.
*   **Multi-Version Support:** Includes RoboTwin 2.0 (latest), RoboTwin 1.0, and early versions, each with unique features and capabilities.
*   **Versatile Task Suite:** Offers a diverse set of tasks, allowing for thorough testing and experimentation.
*   **Policy Baselines:** Provides implementations and support for various policy baselines (DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA).
*   **Leaderboard:**  Track progress and compare results on the official RoboTwin leaderboard.
*   **Extensive Documentation:** Detailed documentation to guide you through installation, usage, and customization.

## Latest Updates

*   **2025/08/28:** RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088) updated.
*   **2025/08/25:**  ACT deployment code fixed, and the [leaderboard](https://robotwin-platform.github.io/leaderboard) updated.
*   **2025/08/06:** RoboTwin 2.0 Leaderboard launched: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **2025/07/23:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/19:** DP3 evaluation code error fixed. RoboTwin 2.0 paper update coming soon.
*   **2025/07/09:** Endpose control mode updated. See the [RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html) for more details.
*   **2025/07/08:**  [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch (第十九届挑战杯分支) uploaded.
*   **2025/07/02:** Piper Wrist Bug fixed [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)]. Please redownload the embodiment asset.
*   **2025/07/01:** Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop released [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21:** RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] released!
*   **2025/04/11:** RoboTwin selected as *CVPR Highlight paper*!
*   **2025/02/27:** RoboTwin accepted to *CVPR 2025*!
*   **2024/09/30:** RoboTwin (Early Version) received *the Best Paper Award at the ECCV Workshop*!
*   **2024/09/20:** Officially released RoboTwin.

## Installation

Detailed installation instructions are available in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).  Installation typically takes around 20 minutes.

## Tasks Information

Explore the available tasks in detail via the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document

Refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for detailed instructions.

### Data Collection

RoboTwin provides a valuable starting point with over 100,000 pre-collected trajectories, accessible within the open-source [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).  However, collecting your own data is highly encouraged, given the platform's high level of customizability and the diverse range of tasks and embodiment setups it supports.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

Run the following command to find a random seed for the target collection quantity, and then replay the seed to collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Modify Task Config

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for details.

## Policy Baselines

### Policies Support

*   DP ([Documentation](https://robotwin-platform.github.io/doc/usage/DP.html))
*   ACT ([Documentation](https://robotwin-platform.github.io/doc/usage/ACT.html))
*   DP3 ([Documentation](https://robotwin-platform.github.io/doc/usage/DP3.html))
*   RDT ([Documentation](https://robotwin-platform.github.io/doc/usage/RDT.html))
*   PI0 ([Documentation](https://robotwin-platform.github.io/doc/usage/Pi0.html))
*   OpenVLA-oft ([Documentation](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html))
*   TinyVLA ([Documentation](https://robotwin-platform.github.io/doc/usage/TinyVLA.html))
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))

Deploy Your Policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, SmolVLA, AVR, UniVLA

## Experiment & Leaderboard

RoboTwin is an ideal platform for exploring:

1.  Single-task fine-tuning
2.  Visual robustness
3.  Language diversity robustness (language condition)
4.  Multi-task capability
5.  Cross-embodiment performance

Find the full leaderboard and settings at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Access the pre-collected dataset at [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you use RoboTwin, please cite the following:

**RoboTwin 2.0:**
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin (CVPR 2025 Highlight):**
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

**RoboTwin Collaboration Challenge at CVPR 2025 MEIS Workshop:**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (ECCV Workshop 2024 Best Paper Award):**
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements

**Software Support:** D-Robotics, **Hardware Support:** AgileX Robotics, **AIGC Support:** Deemos.

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.