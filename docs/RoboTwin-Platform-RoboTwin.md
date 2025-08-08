# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

**RoboTwin is a cutting-edge, open-source platform designed to advance research in dual-arm robotic manipulation, offering a scalable data generator and benchmark for robust, real-world applications.  Explore the RoboTwin platform on [GitHub](https://github.com/RoboTwin-Platform/RoboTwin).**

## Key Features

*   **Scalable Data Generation:** Generate diverse and realistic datasets for training and evaluating bimanual robotic manipulation algorithms.
*   **Strong Domain Randomization:** Enhance the robustness of your models with robust domain randomization techniques.
*   **Comprehensive Benchmark:** Evaluate and compare your algorithms on a challenging set of tasks and scenarios.
*   **Open-Source and Accessible:** Contribute to the community and build upon the foundation of RoboTwin.
*   **Leaderboard:** Track your progress and compete with other researchers on the [RoboTwin Leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **Multiple Versions and Challenges**: Access different versions and challenges (CVPR'25 MEIS Workshop).

## What's New

*   **RoboTwin 2.0 (Latest)**: Introducing a new version with expanded features, improvements and benchmark enhancements.
    *   [Webpage](https://robotwin-platform.github.io/)
    *   [Document](https://robotwin-platform.github.io/doc)
    *   [Paper](https://arxiv.org/abs/2506.18088)
    *   [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop**
    *   Official Technical Report: [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)
*   **Previous Versions**: Explore previous versions for various benchmarks.
    *   RoboTwin 1.0: [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)

## Installation

Follow the detailed installation instructions provided in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation takes approximately 20 minutes.

## Tasks Information

Explore the diverse range of tasks available on RoboTwin, with detailed information in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

For comprehensive usage instructions, consult the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

RoboTwin offers pre-collected trajectories.

### 1. Task Running and Data Collection

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Task Config

Refer to the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for detailed configurations.

## Policy Baselines

RoboTwin supports a variety of policy baselines:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   TinyVLA
*   DexVLA
*   LLaVA-VLA
*   [Deploy Your Policy](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

We recommend that the RoboTwin Platform can be used to explore the following topics:

1.  single-task fine-tuning capability
2.  visual robustness
3.  language diversity robustness (language condition)
4.  multi-tasks capability
5.  cross-embodiment performance

The full leaderboard and settings can be found in: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Citations

If you use RoboTwin in your research, please cite the following papers:

**RoboTwin 2.0**:
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin**:
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

**Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (Early Version)**:
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.

## Contact

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).