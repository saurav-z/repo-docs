# RoboTwin: Revolutionizing Bimanual Robotic Manipulation

RoboTwin is a cutting-edge, open-source platform providing a comprehensive benchmark and data generation tool for advancing research in bimanual robotic manipulation. [Explore the RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate diverse and realistic datasets for robust training and evaluation.
*   **Strong Domain Randomization:**  Enhance generalization capabilities through advanced domain randomization techniques.
*   **Comprehensive Benchmark:** Evaluate and compare your bimanual robotic manipulation algorithms across various tasks.
*   **Pre-collected Large-scale Dataset:** Access and utilize a pre-collected dataset with 100,000+ trajectories on Hugging Face.
*   **Diverse Policies Support:** The platform supports multiple policies, including DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Active Community:** Join the community for support and to share your progress.
*   **Regular Updates:** Stay up-to-date with the latest features, bug fixes, and improvements.
*   **Multiple Versions:** Includes versions with various updates, including the latest RoboTwin 2.0, versions for CVPR and ECCV papers, and challenge-specific branches.

## Latest Updates

*   **2025/08/28:** Updated the RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088).
*   **2025/08/25:** Fixed ACT deployment code and updated the [leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **2025/08/06:** Released RoboTwin 2.0 Leaderboard: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **2025/07/23:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/19:** Fixed DP3 evaluation code error.
*   **2025/07/09:** Updated endpose control mode, please see [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)] for more details.
*   **2025/07/08:** Uploaded [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch (第十九届挑战杯分支).
*   **2025/07/02:** Fixed Piper Wrist Bug [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)].
*   **2025/07/01:** Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21:** Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **2025/04/11:** RoboTwin is selected as *CVPR Highlight paper*!
*   **2025/02/27:** RoboTwin is accepted to *CVPR 2025* !
*   **2024/09/30:** RoboTwin (Early Version) received *the Best Paper Award at the ECCV Workshop*!
*   **2024/09/20:** Officially released RoboTwin.

## Installation

Follow the instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to install RoboTwin. Installation typically takes about 20 minutes.

## Tasks Information

Explore the details of the RoboTwin tasks in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document

Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for detailed information.

### Data Collection

RoboTwin offers a high degree of configurability, so you can collect your own data to maximize control over tasks. The platform also provides pre-collected trajectories.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

Use the following command to run a task and collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Modify Task Config

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for configuration details.

## Policy Baselines

### Policies Supported

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   OpenVLA-oft
*   TinyVLA
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))

Deploy your policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

*⏰ TODO: G3Flow, HybridVLA, SmolVLA, AVR, UniVLA*

## Experiment & Leaderboard

Evaluate your models and compare performance on the [RoboTwin Leaderboard](https://robotwin-platform.github.io/leaderboard).

We recommend the platform be used to explore the following topics:

1.  Single-task fine-tuning capability
2.  Visual robustness
3.  Language diversity robustness (language condition)
4.  Multi-tasks capability
5.  Cross-embodiment performance

## Pre-collected Large-scale Dataset

Find the pre-collected dataset on [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you use RoboTwin in your research, please cite the following:

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

**Benchmarking Generalizable Bimanual Manipulation:**

```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (Early Version):**

```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgement

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

Contact [Tianxing Chen](https://tianxingchen.github.io) with any questions or suggestions.

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.