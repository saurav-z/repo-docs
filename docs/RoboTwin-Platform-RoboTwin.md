# RoboTwin: Revolutionizing Bimanual Robotic Manipulation with Scalable Benchmarking

**RoboTwin** is a cutting-edge platform offering a comprehensive benchmark and data generator for advancing bimanual robotic manipulation, empowering researchers to push the boundaries of robotics; learn more at the [RoboTwin GitHub Repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate massive datasets with strong domain randomization for robust training and evaluation.
*   **Comprehensive Benchmarking:** Evaluate and compare bimanual manipulation algorithms across a variety of tasks and environments.
*   **State-of-the-Art Baselines:** Explore pre-implemented policies like DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA for easy experimentation.
*   **Large-Scale Datasets:** Access over 100,000 pre-collected trajectories through our open-source [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) on Hugging Face.
*   **Active Community & Leaderboard:** Track performance and engage with the community on our [leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **Continuous Updates:** Benefit from regular updates, including new tasks, policies, and improvements.

## Latest Version: RoboTwin 2.0

The newest version, **RoboTwin 2.0**, introduces significant advancements for bimanual robotic manipulation.  

*   **[Webpage](https://robotwin-platform.github.io/)** | **[Document](https://robotwin-platform.github.io/doc)** | **[Paper](https://arxiv.org/abs/2506.18088)** | **[Community](https://robotwin-platform.github.io/doc/community/index.html)** | **[Leaderboard](https://robotwin-platform.github.io/leaderboard)**

## RoboTwin Dual-Arm Collaboration Challenge@CVPR'25 MEIS Workshop

*   Technical Report: [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)

## Previous Versions

### RoboTwin 1.0

*   Accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>: [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)

### RoboTwin (Early Version)

*   Accepted to <i style="color: red; display: inline;"><b>ECCV Workshop 2024 (Best Paper Award)</b></i>: [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Installation

Follow the detailed instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for a quick and easy setup (approx. 20 minutes).

## Tasks & Usage

*   **Tasks:** Explore diverse bimanual manipulation tasks with detailed descriptions at [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).
*   **Usage Documentation:** Comprehensive guides for data collection, task configuration, and policy deployment are available at [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

### Data Collection

Collect your own data tailored to your specific needs and configurations.

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Task Configuration

Customize task configurations; more details can be found at [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines

RoboTwin provides support for various policy baselines to facilitate research and experimentation:

*   DP, ACT, DP3, RDT, PI0, OpenVLA-oft
*   TinyVLA, DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))
*   Deploy Your Policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

Use RoboTwin to investigate:

1.  Single-task fine-tuning.
2.  Visual robustness.
3.  Language diversity robustness.
4.  Multi-task capabilities.
5.  Cross-embodiment performance.

Visit the full leaderboard at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Branches

*   2.0 Version Branch: [main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (latest)
*   1.0 Version Branch: [1.0 Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0)
*   1.0 Version Code Generation Branch: [1.0 Version GPT](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt)
*   Early Version Branch: [Early Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version)
*   第十九届“挑战杯”人工智能专项赛分支: [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025)
*   CVPR 2025 Challenge Round 1 Branch: [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1)
*   CVPR 2025 Challenge Round 2 Branch: [CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2)

## Updates

*   **2025/08/28**: RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088).
*   **2025/08/25**: Fixed ACT deployment code and updated the [leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **2025/08/06**: Released RoboTwin 2.0 Leaderboard: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **2025/07/23**: RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/19**: Fixed DP3 evaluation code error and will update RoboTwin 2.0 paper next week.
*   **2025/07/09**: Updated endpose control mode; more details can be found at [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)].
*   **2025/07/08**: Uploaded [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch (第十九届挑战杯分支).
*   **2025/07/02**: Fixed Piper Wrist Bug [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)]. Please redownload the embodiment asset.
*   **2025/07/01**: Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21**: Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **2025/04/11**: RoboTwin is selected as *CVPR Highlight paper*!
*   **2025/02/27**: RoboTwin is accepted to *CVPR 2025* !
*   **2024/09/30**: RoboTwin (Early Version) received *the Best Paper Award at the ECCV Workshop*!
*   **2024/09/20**: Officially released RoboTwin.

## Acknowledgements

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos.

Contact [Tianxing Chen](https://tianxingchen.github.io) for any questions or suggestions.

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

## License

This repository is released under the MIT license; see [LICENSE](./LICENSE) for details.