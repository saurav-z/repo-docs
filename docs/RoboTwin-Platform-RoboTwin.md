# RoboTwin: Revolutionizing Bimanual Robotic Manipulation

**RoboTwin is a cutting-edge platform for bimanual robotic manipulation research, providing a scalable data generator, benchmark, and challenge, built with strong domain randomization for robust performance. Explore the platform on [GitHub](https://github.com/RoboTwin-Platform/RoboTwin)**

## Key Features:

*   **Scalable Data Generation:** Generate massive datasets for training and evaluation.
*   **Robust Domain Randomization:** Enhance the generalization capabilities of your robotic models.
*   **Comprehensive Benchmark:** Evaluate your algorithms on a wide range of bimanual manipulation tasks.
*   **Challenge & Leaderboard:** Compete and compare your models with the state-of-the-art.
*   **Multiple Task Support:** Includes 50+ tasks, covering a wide range of bimanual manipulation skills.
*   **Various Policy Baseline:** DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA baselines.

## Latest Updates:

*   **RoboTwin 2.0 Leaderboard:** Access the updated leaderboard at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **RoboTwin 2.0 Released!** Explore the newest version with enhanced features [Webpage](https://robotwin-platform.github.io/)!

## RoboTwin Versions:

*   **RoboTwin 2.0:** (Latest) A Scalable Data Generator and Benchmark with Strong Domain Randomization (arXiv:2506.18088)
    *   [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Community](https://robotwin-platform.github.io/doc/community/index.html) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** Technical Report (arXiv:2506.23351)
*   **RoboTwin 1.0:** Dual-Arm Robot Benchmark with Generative Digital Twins (CVPR 2025 Highlight) (arXiv:2504.13059)
*   **RoboTwin (Early Version):** Dual-Arm Robot Benchmark with Generative Digital Twins (ECCV Workshop 2024 Best Paper Award) (arXiv:2409.02920)

## Installation:

Detailed installation instructions are available in the RoboTwin 2.0 document: [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).  Installation typically takes about 20 minutes.

## Tasks Information:

Explore the diverse range of tasks supported by RoboTwin: [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage:

### Document:

Refer to the official documentation for comprehensive usage details: [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html)

### Data Collection:

*   Pre-collected trajectories are available in the [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
*   We strongly recommend you to collect your own data.

<img src="./assets/files/domain_randomization.png" alt="Domain Randomization" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Modify Task Config:

Customize task configurations to your needs: [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html)

## Policy Baselines:

Explore the policy baselines, including:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   OpenVLA-oft
*   TinyVLA
*   DexVLA
*   LLaVA-VLA

Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard:

Explore the platform for the following topics:

*   Single - task fine - tuning capability
*   Visual robustness
*   Language diversity robustness (language condition)
*   Multi-tasks capability
*   Cross-embodiment performance

Find the full leaderboard and settings at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset:

Access the pre-collected dataset on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations:

If you utilize RoboTwin in your research, please cite the following papers:

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

## Acknowledgements:

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License:

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.