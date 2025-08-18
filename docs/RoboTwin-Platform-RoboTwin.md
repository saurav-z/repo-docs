<h1 align="center">
  RoboTwin: The Premier Bimanual Robotic Manipulation Platform
</h1>

<p align="center">
  <a href="https://github.com/RoboTwin-Platform/RoboTwin">
    <img src="https://img.shields.io/github/stars/RoboTwin-Platform/RoboTwin?style=social" alt="Stars">
  </a>
  <a href="https://robotwin-platform.github.io/">
    <img src="https://img.shields.io/website?label=Website&url=https%3A%2F%2Frobotwin-platform.github.io%2F" alt="Website">
  </a>
  <a href="https://robotwin-platform.github.io/doc/">
    <img src="https://img.shields.io/badge/Documentation-View-blue" alt="Documentation">
  </a>
  <a href="https://robotwin-platform.github.io/leaderboard">
    <img src="https://img.shields.io/badge/Leaderboard-View-green" alt="Leaderboard">
  </a>
</p>

## Dive into the future of robotics with RoboTwin, a cutting-edge platform for bimanual robotic manipulation.

RoboTwin provides a comprehensive suite for research and development in dual-arm robotics, featuring a scalable data generator, robust domain randomization, and a competitive benchmark.

**Key Features:**

*   🎯 **Scalable Data Generation:** Generate diverse datasets for training and evaluation.
*   ⚙️ **Strong Domain Randomization:** Enhance the robustness of your models against real-world variations.
*   🏆 **Comprehensive Benchmark:** Evaluate and compare your algorithms on a challenging set of tasks.
*   🤖 **Latest Version 2.0:** Explore the newest advancements in bimanual robotic manipulation.
*   📚 **Extensive Documentation:** Detailed documentation to help you get started quickly.
*   🚀 **Leaderboard & Community:**  Track performance and collaborate with other researchers.

## RoboTwin Versions

*   **RoboTwin 2.0 (Latest)**: [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge@CVPR'25 MEIS Workshop** : [Technical Report (PDF)](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0:** [CVPR 2025 (Highlight) Paper (PDF)](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **RoboTwin (Early Version):** [ECCV Workshop 2024 (Best Paper Award) Paper (PDF)](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Quick Links

*   [**Installation**](https://robotwin-platform.github.io/doc/usage/robotwin-install.html): Installation Guide
*   [**Tasks Information**](https://robotwin-platform.github.io/doc/tasks/index.html): Explore available tasks.
*   [**Usage**](https://robotwin-platform.github.io/doc/usage/index.html): Getting Started Guide.
*   [**Policy Baselines**](https://robotwin-platform.github.io/doc/usage/index.html): Supported Policies

## Installation

Follow the instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) (Usage - Install & Download) to set up your environment. The installation process typically takes about 20 minutes.

## Tasks

RoboTwin offers a variety of challenging tasks to test your bimanual robotic manipulation algorithms. Find details in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

Comprehensive usage instructions can be found in the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

Collect your own data to explore the high configurability and task diversity.

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Task Configuration

Customize your experiments using the configurations detailed in the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines

RoboTwin supports a range of policy baselines:

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html)
*   [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
*   [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
*   [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
*   [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
*   [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)
*   [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) (Contributed by IRPN Lab, HKUST(GZ))

Learn how to [deploy your own policy](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## Experiment & Leaderboard

Explore the RoboTwin Platform to test:

1.  Single-task fine-tuning capability.
2.  Visual robustness.
3.  Language diversity robustness (language condition).
4.  Multi-task capability.
5.  Cross-embodiment performance.

Find the full leaderboard at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Citations

If you find RoboTwin useful, please cite the following:

```bibtex
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}

@InProceedings{Mu_2025_CVPR,
    author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
    title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27649-27660}
}

@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}

@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos

Code Style: `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

For questions or suggestions, contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT license. See [LICENSE](./LICENSE) for details.

[**Go back to the RoboTwin repository.**](https://github.com/RoboTwin-Platform/RoboTwin)