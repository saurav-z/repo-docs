# RoboTwin: Revolutionizing Bimanual Robotic Manipulation 🤖

RoboTwin is a cutting-edge platform providing a comprehensive benchmark for bimanual robotic manipulation, including a scalable data generator, robust domain randomization, and a suite of pre-built tasks, detailed [here](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate diverse and realistic datasets for training and evaluating your bimanual robotic manipulation algorithms.
*   **Strong Domain Randomization:**  Enhance the robustness of your models through rigorous domain randomization techniques.
*   **Comprehensive Benchmark:**  Evaluate your models against a diverse set of pre-defined bimanual robotic manipulation tasks.
*   **Open-Source and Accessible:**  Leverage the power of RoboTwin with its open-source nature and easy-to-follow documentation.
*   **Pre-collected Trajectories:** Access over 100,000 pre-collected trajectories to kickstart your research.
*   **Multiple Policy Baselines:** Utilize and compare performance with established baselines, including DP, ACT, DP3, RDT, PI0, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Active Community:** Join the vibrant community and find detailed installation guides, task information, and usage instructions.

## RoboTwin Versions & Updates

*   **RoboTwin 2.0 (Latest)**:  A major update featuring a scalable data generator and benchmark with robust domain randomization.
    *   [Webpage](https://robotwin-platform.github.io/) | [Documentation](https://robotwin-platform.github.io/doc) | [Paper (arXiv)](https://arxiv.org/abs/2506.18088)
*   **RoboTwin Dual-Arm Collaboration Challenge@CVPR'25 MEIS Workshop:** Technical Report available.
    *   [Technical Report (arXiv)](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0**:  The original benchmark with generative digital twins.
    *   Accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>: [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **Early Version:** The foundational version of RoboTwin.
    *   Accepted to <i style="color: red; display: inline;"><b>ECCV Workshop 2024 (Best Paper Award)</b></i>: [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## 🛠️ Installation

Follow the detailed installation instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes around 20 minutes.

## 🤷‍♂️ Tasks Information

Explore the comprehensive tasks available on the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

## 🧑🏻‍💻 Usage

For in-depth usage instructions, refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

## 🚴‍♂️ Policy Baselines

RoboTwin supports various policy baselines:

*   DP, ACT, DP3, RDT, PI0
*   TinyVLA, DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))
*   [Deploy Your Policy: guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## 🏄‍♂️ Experiment & Leaderboard

(Coming Soon: Instructions for experimentation and the Leaderboard)

## 👍 Citations

If you utilize RoboTwin in your work, please cite the relevant papers:

**RoboTwin 2.0:**
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
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

**RoboTwin (Early Version):**
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## 😺 Acknowledgements

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos

## 🏷️ License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).