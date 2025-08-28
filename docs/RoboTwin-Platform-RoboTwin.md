# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

**RoboTwin** is a cutting-edge platform designed to advance the field of bimanual robotic manipulation, offering a comprehensive suite of tools for research, development, and benchmarking.  Explore the latest advancements and contribute to the future of robotics!  [Go to the original repository](https://github.com/RoboTwin-Platform/RoboTwin)

## Key Features

*   **Robust Domain Randomization:**  Simulate diverse environments to enhance the generalizability of robotic skills.
*   **Scalable Data Generation:** Generate large-scale datasets for training and evaluating robotic models efficiently.
*   **Comprehensive Benchmarking:** Evaluate the performance of various algorithms on a range of bimanual manipulation tasks.
*   **Multiple Policy Baselines:** Supports DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA and LLaVA-VLA, and more.
*   **Leaderboard:** Track and compare your results with other researchers on our official leaderboard.
*   **Active Community:** Access documentation, a community forum, and collaboration opportunities.

##  Latest Updates

*   **[2025/08/28]** RoboTwin 2.0 Paper updated [PDF](https://arxiv.org/pdf/2506.18088).
*   **[2025/08/25]** Fix ACT deployment code and leaderboard update.
*   **[2025/08/06]** RoboTwin 2.0 Leaderboard: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **[2025/07/23]** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **[2025/07/19]** Fix DP3 evaluation code error.  RoboTwin 2.0 paper will be updated next week.
*   **[2025/07/09]** Updated endpose control mode [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)] for details.
*   **[2025/07/08]** Uploaded [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch (第十九届挑战杯分支).
*   **[2025/07/02]** Fix Piper Wrist Bug [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)]. Please redownload the embodiment asset.
*   **[2025/07/01]** Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **[2025/06/21]** Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **[2025/04/11]** RoboTwin is seclected as *CVPR Highlight paper*!
*   **[2025/02/27]** RoboTwin is accepted to *CVPR 2025* !
*   **[2024/09/30]** RoboTwin (Early Version) received *the Best Paper Award at the ECCV Workshop*!
*   **[2024/09/20]** Officially released RoboTwin.

## Versions & Resources

*   **RoboTwin 2.0 (Latest):** [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Community](https://robotwin-platform.github.io/doc/community/index.html) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR '25 MEIS Workshop:** [Technical Report (PDF)](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0:** [Paper (CVPR 2025 Highlight)](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **RoboTwin (Early Version):** [Paper (ECCV Workshop Best Paper)](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Branches

| Branch Name                      | Link                                                                           |
| :------------------------------- | :----------------------------------------------------------------------------- |
| 2.0 Version (main)               | [main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (latest)          |
| 1.0 Version                    | [1.0 Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0)    |
| 1.0 Version Code Generation  | [1.0 Version GPT](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt)           |
| Early Version                    | [Early Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version) |
| 挑战杯专项赛分支           | [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) |
| CVPR 2025 Challenge Round 1   | [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1)   |
| CVPR 2025 Challenge Round 2   | [CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2)   |

## Installation

Follow the detailed installation instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes around 20 minutes.

## Tasks Overview

Explore the diverse range of tasks supported by RoboTwin: [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Documentation

Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for comprehensive details.

### Data Collection

We provide over 100,000 pre-collected trajectories [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).  However, we recommend performing your own data collection for task configurability and diversity.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### Task Execution and Data Collection

Run the following command to collect data for a specific task:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Task Configuration

Customize tasks using detailed configuration options: [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines

### Supported Policies

*   DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA (Contributed by Media Group), LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))
*   Deploy Your Policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)
*   *Coming Soon:* G3Flow, HybridVLA, SmolVLA, AVR, UniVLA

## Experiment & Leaderboard

Explore the potential of RoboTwin for:
*   single - task fine - tuning
*   visual robustness
*   language diversity robustness (language condition)
*   multi-tasks capability
*   cross-embodiment performance

View the full leaderboard and settings: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Access the pre-collected dataset: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

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

## Acknowledgements

**Software Support:** D-Robotics, **Hardware Support:** AgileX Robotics, **AIGC Support:** Deemos.

For questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT license. See [LICENSE](./LICENSE) for details.