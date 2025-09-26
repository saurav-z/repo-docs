# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

RoboTwin is a cutting-edge platform designed to advance the field of bimanual robotic manipulation through scalable data generation, strong domain randomization, and challenging benchmark tasks. [Explore the RoboTwin Repository](https://github.com/RoboTwin-Platform/RoboTwin) to unlock new possibilities in robotics.

## Key Features

*   **Scalable Data Generation:** Generate massive datasets for training robust robotic models.
*   **Strong Domain Randomization:** Overcome the reality gap with advanced domain randomization techniques.
*   **Challenging Benchmark Tasks:** Evaluate and compare algorithms on diverse and complex bimanual manipulation tasks.
*   **Pre-collected Datasets:** Access a large-scale pre-collected dataset on Hugging Face.
*   **Policy Baselines:** Implement and test your models with a diverse set of policy baselines, including DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Community Support:** Engage with the RoboTwin community through the official [community page](https://robotwin-platform.github.io/doc/community/index.html).
*   **Leaderboard:** Track progress and compare performance on the [RoboTwin Leaderboard](https://robotwin-platform.github.io/leaderboard).

## Recent Updates

*   **2025/08/28:** Updated RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088).
*   **2025/08/25:** Fixed ACT deployment code and updated the [leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **2025/08/06:** Released RoboTwin 2.0 Leaderboard: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **2025/07/23:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/19:** Fixed DP3 evaluation code error. Will update RoboTwin 2.0 paper next week.
*   **2025/07/09:** Updated endpose control mode, see [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)] for more details.
*   **2025/07/08:** Uploaded [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch.
*   **2025/07/02:** Fixed Piper Wrist Bug [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)]. Please redownload the embodiment asset.
*   **2025/07/01:** Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21:** Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **2025/04/11:** RoboTwin is selected as *CVPR Highlight paper*!
*   **2025/02/27:** RoboTwin is accepted to *CVPR 2025*!
*   **2024/09/30:** RoboTwin (Early Version) received *the Best Paper Award at the ECCV Workshop*!
*   **2024/09/20:** Officially released RoboTwin.

## Installation

Refer to the detailed installation instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes around 20 minutes.

## Tasks

Explore the diverse range of bimanual manipulation tasks available in RoboTwin. For detailed task information, please refer to the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document

For comprehensive usage guidelines, consult the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

We provide over 100,000 pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
We strongly recommend users to perform data collection themselves due to the high configurability and diversity of task and embodiment setups.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

Run the following command to collect data, first searching for a random seed for the target collection quantity, and then replaying the seed to collect data.

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Modify Task Config

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

### Deploy Your Policy

See the deployment guidance: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, SmolVLA, AVR, UniVLA

## Experiment & Leaderboard

We encourage users to explore the following areas using the RoboTwin Platform:

1.  Single-task fine-tuning capability.
2.  Visual robustness.
3.  Language diversity robustness (language condition).
4.  Multi-task capability.
5.  Cross-embodiment performance.

The full leaderboard and settings can be found at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Access the pre-collected dataset on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you find RoboTwin useful, please cite our work:

**RoboTwin 2.0:** A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation

```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin:** Dual-Arm Robot Benchmark with Generative Digital Twins, accepted to <i style="color: red; display: inline;">CVPR 2025 (Highlight)</i>

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

**RoboTwin:** Dual-Arm Robot Benchmark with Generative Digital Twins (early version), accepted to <i style="color: red; display: inline;">ECCV Workshop 2024 (Best Paper Award)</i>

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

For questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.