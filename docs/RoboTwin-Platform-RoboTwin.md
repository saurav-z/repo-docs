# RoboTwin: A Scalable Benchmark for Bimanual Robotic Manipulation

RoboTwin provides a comprehensive and scalable platform for research in bimanual robotic manipulation, featuring strong domain randomization and a suite of challenging tasks. Explore the project on [GitHub](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Scalable Data Generation:** Efficiently generate large-scale datasets for training and evaluating robotic manipulation algorithms.
*   **Strong Domain Randomization:** Built-in domain randomization techniques to improve the robustness of robotic manipulation models.
*   **Challenging Benchmark Tasks:**  Includes a diverse set of tasks to test and benchmark bimanual robotic manipulation skills, including [50 Tasks](https://robotwin-platform.github.io/doc/tasks/index.html).
*   **Policy Baselines:** Support for various policies, including DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA, with more to come.
*   **Pre-collected Datasets:** Access a large pre-collected dataset of over 100,000 trajectories via [Hugging Face](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
*   **Leaderboard:**  Track and compare your model's performance on the RoboTwin leaderboard: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **Comprehensive Documentation:** Detailed documentation for installation, usage, and task information: [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/index.html)

## What's New

*   **RoboTwin 2.0 is out!** Featuring a new [Webpage](https://robotwin-platform.github.io/) and [Paper](https://arxiv.org/pdf/2506.18088), with a new [Leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **CVPR 2025 Highlight Paper & CVPR 2025 Dual-Arm Collaboration Challenge at MEIS Workshop**
*   **Best Paper Award at ECCV Workshop 2024**

## Installation

Follow the detailed installation instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).  Installation typically takes about 20 minutes.

## Usage

### Document

> Please Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for more details.

### Data Collection

We provide over 100,000 pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
However, we strongly recommend users to perform data collection themselves due to the high configurability and diversity of task and embodiment setups.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

Running the following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Modify Task Config

☝️ See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

## Policy Baselines

### Policies Support

[DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html)

[TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)

[LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) (Contributed by IRPN Lab, HKUST(GZ))

Deploy Your Policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, SmolVLA, AVR, UniVLA

## Experiment & LeaderBoard

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

**RoboTwin**: Dual-Arm Robot Benchmark with Generative Digital Twins, accepted to <i style="color: red; display: inline;">**CVPR 2025 (Highlight)***

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

**RoboTwin**: Dual-Arm Robot Benchmark with Generative Digital Twins (early version), accepted to <i style="color: red; display: inline;">**ECCV Workshop 2024 (Best Paper Award)***

```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos.

Contact [Tianxing Chen](https://tianxingchen.github.io) for questions or suggestions.

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.