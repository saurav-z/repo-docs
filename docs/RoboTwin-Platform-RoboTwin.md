# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

**RoboTwin** is a cutting-edge, open-source platform designed to benchmark and advance the field of bimanual robotic manipulation, offering a scalable data generator and benchmark with strong domain randomization for robust performance. Explore the platform at the [RoboTwin GitHub Repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Advanced Benchmark:** Evaluate and compare bimanual robotic manipulation algorithms.
*   **Scalable Data Generation:** Generate diverse and realistic datasets with domain randomization.
*   **Robust Domain Randomization:** Enhance the generalizability of robotic skills.
*   **Multiple Baselines:**  Includes support for policies like DP, ACT, DP3, RDT, PI0, and OpenVLA-oft.
*   **Open Source:**  Freely available under the MIT license, fostering collaboration and innovation.
*   **Leaderboard:**  Track your progress and compare with others in the RoboTwin community.
*   **Extensive Documentation:** Detailed documentation for installation, usage, and task information.

## What's New:

*   **RoboTwin 2.0:** The latest version with enhanced features and improvements. ([Webpage](https://robotwin-platform.github.io/), [Document](https://robotwin-platform.github.io/doc), [Paper](https://arxiv.org/abs/2506.18088), [Leaderboard](https://robotwin-platform.github.io/leaderboard))
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** Participate in the ongoing challenge and contribute to advancing the field. ([Technical Report](https://arxiv.org/abs/2506.23351))

## Overview:

*   **RoboTwin 2.0:** [main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (latest)
*   **RoboTwin 1.0:** [1.0 Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0)
*   **RoboTwin 1.0 Code Generation:** [1.0 Version GPT](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt)
*   **RoboTwin Early Version:** [Early Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version)
*   **Challenge Cup 2025:** [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025)
*   **CVPR 2025 Challenge Round 1:** [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1)
*   **CVPR 2025 Challenge Round 2:** [CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2)

## Updates:

*   **2025/08/25:** Fixed ACT deployment code and updated the [leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **2025/08/06:** Released RoboTwin 2.0 Leaderboard.
*   **2025/07/23:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025.
*   **2025/07/19:** Fixed DP3 evaluation code error and announced RoboTwin 2.0 paper update.
*   **2025/07/09:** Updated endpose control mode; see [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)].
*   **2025/07/08:** Uploaded [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch.
*   **2025/07/02:** Fixed Piper Wrist Bug.
*   **2025/07/01:** Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)].
*   **2025/06/21:** Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)].
*   **2025/04/11:** RoboTwin is selected as <i>CVPR Highlight paper</i>.
*   **2025/02/27:** RoboTwin is accepted to <i>CVPR 2025</i>.
*   **2024/09/30:** RoboTwin (Early Version) received the Best Paper Award at the ECCV Workshop.
*   **2024/09/20:** Officially released RoboTwin.

## Installation:

Follow the instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation takes approximately 20 minutes.

## Tasks:

Explore a wide variety of tasks.  See the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage:

### Document

Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for detailed usage instructions.

### Data Collection

Easily collect data with the high configurability and diversity of task and embodiment setups.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### Task Running and Data Collection

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Modify Task Config

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines:

Support for various policies, including:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   OpenVLA-oft
*   TinyVLA
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))

Deploy your own policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard:

Evaluate your models and compete on the [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

> Recommended topics:
>
> 1.  Single-task fine-tuning capability
> 2.  Visual robustness
> 3.  Language diversity robustness
> 4.  Multi-task capability
> 5.  Cross-embodiment performance

## Pre-collected Dataset:

Explore the [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations:

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

## Acknowledgement:

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos.

Contact [Tianxing Chen](https://tianxingchen.github.io) with any questions or suggestions.

## License:

This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.