# RoboTwin: Advancing Bimanual Robotic Manipulation with a Scalable Benchmark

RoboTwin is a cutting-edge platform offering a comprehensive benchmark and data generator for bimanual robotic manipulation, empowering researchers to develop robust and generalizable robotic solutions.  Explore the RoboTwin platform on [GitHub](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate diverse and realistic training data with strong domain randomization for robust performance.
*   **Comprehensive Benchmark:** Evaluate your bimanual robotic manipulation algorithms across a variety of challenging tasks.
*   **Open-Source and Accessible:** Benefit from a freely available platform, facilitating research and collaboration.
*   **Multiple Versions & Challenges:** Includes RoboTwin 2.0, RoboTwin Dual-Arm Collaboration Challenge and earlier versions, each with unique features.
*   **Leaderboard:** Track and compare your results against state-of-the-art models on the RoboTwin leaderboard.
*   **Pre-collected Large-scale Dataset:** Access over 100,000 pre-collected trajectories to kickstart your research.
*   **Multiple Baseline Policies:** Implement your policy using DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA and LLaVA-VLA.
*   **Extensive Documentation:** Access comprehensive documentation and guides to streamline your research.

## Latest Updates

*   **2025/08/28:** Updated RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088).
*   **2025/08/25:** Fixed ACT deployment code and updated the [leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **2025/08/06:** Released RoboTwin 2.0 Leaderboard: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **2025/07/23:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/19:** Fixed DP3 evaluation code error.
*   **2025/07/09:** Updated endpose control mode, see [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)] for more details.
*   **2025/07/08:** Uploaded [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch.
*   **2025/07/02:** Fixed Piper Wrist Bug. Please redownload the embodiment asset.
*   **2025/07/01:** Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21:** Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
*   **2025/04/11:** RoboTwin is selected as *CVPR Highlight paper*!
*   **2025/02/27:** RoboTwin is accepted to *CVPR 2025* !
*   **2024/09/30:** RoboTwin (Early Version) received the Best Paper Award at the ECCV Workshop!
*   **2024/09/20:** Officially released RoboTwin.

## Installation

Follow the instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation takes approximately 20 minutes.

## Tasks Information

For detailed information on available tasks, refer to the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document

See [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for detailed instructions.

### Data Collection

We provide a large dataset of pre-collected trajectories. We recommend creating your own data due to the high configurability and diversity.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Modify Task Config

Refer to [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) to customize your experiments.

## Policy Baselines

### Policies Supported

*   DP ([Documentation](https://robotwin-platform.github.io/doc/usage/DP.html))
*   ACT ([Documentation](https://robotwin-platform.github.io/doc/usage/ACT.html))
*   DP3 ([Documentation](https://robotwin-platform.github.io/doc/usage/DP3.html))
*   RDT ([Documentation](https://robotwin-platform.github.io/doc/usage/RDT.html))
*   PI0 ([Documentation](https://robotwin-platform.github.io/doc/usage/Pi0.html))
*   OpenVLA-oft ([Documentation](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html))
*   TinyVLA ([Documentation](https://robotwin-platform.github.io/doc/usage/TinyVLA.html))
*   DexVLA ([Documentation](https://robotwin-platform.github.io/doc/usage/DexVLA.html)) (Contributed by Media Group)
*   LLaVA-VLA ([Documentation](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html)) (Contributed by IRPN Lab, HKUST(GZ))

### Deploy Your Policy

Follow the [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html) to deploy your own policy.

## Experiment & Leaderboard

We recommend using RoboTwin for exploring:

1.  Single-task fine-tuning.
2.  Visual robustness.
3.  Language diversity robustness (language condition).
4.  Multi-task capability.
5.  Cross-embodiment performance.

Find the full leaderboard and settings at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Access the dataset on Huggingface: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you use RoboTwin, please cite the following:

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

## Acknowledgements

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

Contact [Tianxing Chen](https://tianxingchen.github.io) with any questions or suggestions.

## License

This repository is released under the MIT License. See [LICENSE](./LICENSE) for more information.