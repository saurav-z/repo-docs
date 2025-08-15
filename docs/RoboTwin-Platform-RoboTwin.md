# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

**Explore the future of robotics with RoboTwin, a cutting-edge platform for advancing bimanual robotic manipulation, and available on [GitHub](https://github.com/RoboTwin-Platform/RoboTwin).**

## Key Features

*   **Comprehensive Benchmark:** RoboTwin provides a robust benchmark for evaluating bimanual robotic manipulation algorithms across a wide range of tasks.
*   **Scalable Data Generation:** Includes a scalable data generator that enables efficient training and evaluation of models.
*   **Strong Domain Randomization:** Leverages strong domain randomization techniques to enhance the robustness and generalizability of robotic manipulation systems.
*   **Multiple Versions & Leaderboard:** Access to various RoboTwin versions (2.0, 1.0, early versions) and a public leaderboard to track progress and compare performance.
*   **Extensive Documentation and Resources:** Detailed documentation, tutorials, and code examples to help you get started quickly.
*   **Community Support:** Access to a community for discussions, collaboration, and support.

## What's New in RoboTwin 2.0

*   **RoboTwin 2.0:** [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [arXiv](https://arxiv.org/abs/2506.18088) | [Talk (in Chinese)](https://www.bilibili.com/video/BV18p3izYE63/?spm_id_from=333.337.search-card.all.click) | [机器之心](https://mp.weixin.qq.com/s/SwORezmol2Qd9YdrGYchEA) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)

    *   RoboTwin 2.0 is a Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation.

## Key Updates

*   **Latest News:**
    *   **2025/08/06**, RoboTwin 2.0 Leaderboard is released: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
    *   **2025/07/23**, RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
    *   **2025/07/19**, We Fix DP3 evaluation code error. We will update RoboTwin 2.0 paper next week.
    *   **2025/07/09**, Update endpose control mode.
    *   **2025/07/08**, Upload [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch.
    *   **2025/07/02**, Fix Piper Wrist Bug [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)]. Please redownload the embodiment asset.
    *   **2025/07/01**, Release Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
    *   **2025/06/21**, Release RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
    *   **2025/04/11**, RoboTwin is selected as *CVPR Highlight paper*!
    *   **2025/02/27**, RoboTwin is accepted to *CVPR 2025*!
    *   **2024/09/30**, RoboTwin (Early Version) received *the Best Paper Award at the ECCV Workshop*!
    *   **2024/09/20**, Officially released RoboTwin.

*   **RoboTwin Dual-Arm Collaboration Challenge@CVPR'25 MEIS Workshop:**
    *   Official Technical Report: [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351) | [量子位](https://mp.weixin.qq.com/s/qxqs9vvvHsAJ-0hoYANYzQ)

*   **RoboTwin 1.0:**
    *   Accepted to *CVPR 2025 (Highlight)*: [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)

*   **Early Version:**
    *   Accepted to *ECCV Workshop 2024 (Best Paper Award)*: [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Installation

See [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for installation instructions. Installation takes approximately 20 minutes.

## Tasks & Usage

### Tasks Information

See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.
<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

### Data Collection

We provide over 100,000 pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
However, we strongly recommend users to perform data collection themselves due to the high configurability and diversity of task and embodiment setups.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### Task Running and Data Collection

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Task Configuration

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

## Policy Baselines

### Policies Supported

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html)
*   [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
*   [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
*   [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
*   [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
*   [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)
*   [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) (Contributed by IRPN Lab, HKUST(GZ))

Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

We recommend that the RoboTwin Platform can be used to explore the following topics:

*   single-task fine-tuning capability
*   visual robustness
*   language diversity robustness (language condition)
*   multi-tasks capability
*   cross-embodiment performance

The full leaderboard and setting can be found in: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Citations

If you find our work useful, please consider citing:

**RoboTwin 2.0:**
```bibtex
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin 1.0:**
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

**Benchmarking Generalizable Bimanual Manipulation:**
```bibtex
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (Early Version):**
```bibtex
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements

*   **Software Support:** D-Robotics
*   **Hardware Support:** AgileX Robotics
*   **AIGC Support:** Deemos

Code Style: `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

Contact [Tianxing Chen](https://tianxingchen.github.io) for questions and suggestions.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.