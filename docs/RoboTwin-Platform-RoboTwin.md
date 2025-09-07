# RoboTwin: A Powerful Benchmark for Bimanual Robotic Manipulation

**RoboTwin** is a comprehensive platform designed to advance research in bimanual robotic manipulation, providing a scalable data generator, a robust benchmark, and strong domain randomization. Learn more at the [RoboTwin GitHub repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate large-scale datasets to train and evaluate your robotic manipulation models.
*   **Robust Domain Randomization:** Implement strong domain randomization techniques for improved generalization and real-world performance.
*   **Comprehensive Benchmark:** Evaluate your algorithms on a diverse set of manipulation tasks.
*   **Multiple Versions & Challenges:** Explore various versions of RoboTwin, including the latest RoboTwin 2.0 and participate in challenges like the RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop.
*   **Open-Source & Community Driven:** Benefit from an active community, documentation, and open-source code.
*   **Leaderboard & Evaluation:** Track your progress and compare your results on the RoboTwin Leaderboard.

## RoboTwin Versions

*   **RoboTwin 2.0 (Latest):** A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation ([Webpage](https://robotwin-platform.github.io/), [Document](https://robotwin-platform.github.io/doc), [Paper](https://arxiv.org/abs/2506.18088), [Leaderboard](https://robotwin-platform.github.io/leaderboard))
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** Technical Report ([PDF](https://arxiv.org/pdf/2506.23351), [arXiv](https://arxiv.org/abs/2506.23351))
*   **RoboTwin 1.0:** Dual-Arm Robot Benchmark with Generative Digital Twins ([PDF](https://arxiv.org/pdf/2504.13059), [arXiv](https://arxiv.org/abs/2504.13059)) - _CVPR 2025 Highlight_
*   **RoboTwin (Early Version):** Dual-Arm Robot Benchmark with Generative Digital Twins ([PDF](https://arxiv.org/pdf/2409.02920), [arXiv](https://arxiv.org/abs/2409.02920)) - _ECCV Workshop 2024 Best Paper Award_

## Updates

*   **2025/08/28:** Updated RoboTwin 2.0 Paper ([PDF](https://arxiv.org/pdf/2506.18088)).
*   **2025/08/25:** Fixed ACT deployment code and updated the [leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **2025/08/06:** Released RoboTwin 2.0 Leaderboard: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **2025/07/23:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/19:** Fixed DP3 evaluation code error.
*   **2025/07/09:** Updated endpose control mode.
*   **2025/07/08:** Uploaded [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch.
*   **2025/07/02:** Fixed Piper Wrist Bug.
*   **2025/07/01:** Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21:** Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **2025/04/11:** RoboTwin is seclected as _CVPR Highlight paper_!
*   **2025/02/27:** RoboTwin is accepted to _CVPR 2025_ !
*   **2024/09/30:** RoboTwin (Early Version) received _the Best Paper Award at the ECCV Workshop_!
*   **2024/09/20:** Officially released RoboTwin.

## Installation

Detailed installation instructions can be found in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation takes approximately 20 minutes.

## Tasks Information

See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document

> Please Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for more details.

### Data Collection

We provide over 100,000 pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
However, we strongly recommend users to perform data collection themselves due to the high configurability and diversity of task and embodiment setups.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### Task Running and Data Collection

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Modify Task Config

☝️ See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

## Policy Baselines

### Policies Supported

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html)
*   [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
*   [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
*   [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
*   [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
*   [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)
*   [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) (Contributed by IRPN Lab, HKUST(GZ))

Deploy Your Policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, SmolVLA, AVR, UniVLA

## Experiment & Leaderboard

> We recommend that the RoboTwin Platform can be used to explore the following topics:
> 1. single - task fine - tuning capability
> 2. visual robustness
> 3. language diversity robustness (language condition)
> 4. multi-tasks capability
> 5. cross-embodiment performance

The full leaderboard and setting can be found in: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Please refer to [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you find our work useful, please consider citing:

**RoboTwin 2.0**: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation

```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin**: Dual-Arm Robot Benchmark with Generative Digital Twins, accepted to <i style="color: red; display: inline;">**CVPR 2025 (Highlight)**</i>

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

**RoboTwin**: Dual-Arm Robot Benchmark with Generative Digital Twins (early version), accepted to <i style="color: red; display: inline;">**ECCV Workshop 2024 (Best Paper Award)**</i>

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

Contact [Tianxing Chen](https://tianxingchen.github.io) if you have any questions or suggestions.

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.