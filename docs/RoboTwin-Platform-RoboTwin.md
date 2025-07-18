<h1 align="center">
  <a href="https://robotwin-benchmark.github.io"><b>RoboTwin: The Premier Bimanual Robotic Manipulation Platform</b></a>
</h1>

<p align="center">
  <a href="https://github.com/RoboTwin-Platform/RoboTwin">
    <img src="https://img.shields.io/github/stars/RoboTwin-Platform/RoboTwin?style=social" alt="Stars">
  </a>
</p>

<h2 align="center">
  Explore the cutting edge of bimanual robotic manipulation with RoboTwin 2.0! 🤲
  <br>
  <a href="https://robotwin-platform.github.io/">Webpage</a> | <a href="https://robotwin-platform.github.io/doc/">Documentation</a> | <a href="https://arxiv.org/abs/2506.18088">Paper</a> | <a href="https://robotwin-platform.github.io/doc/community/index.html">Community</a>
</h2>

[![RoboTwin Demo](https://private-user-images.githubusercontent.com/88101805/463126988-e3ba1575-4411-4a36-ad65-f0b2f49890c3.mp4)](https://github.com/RoboTwin-Platform/RoboTwin)

RoboTwin is a comprehensive platform for research and development in bimanual robotic manipulation, offering a scalable data generator, robust domain randomization, and a suite of pre-built tasks and baselines.

Key Features:

*   **RoboTwin 2.0:** The latest version features enhanced scalability and improved domain randomization.
*   **Benchmark:** A robust benchmark for bimanual robotic manipulation.
*   **Data Generation:** Scalable data generation capabilities for training and evaluation.
*   **Domain Randomization:** Strong domain randomization techniques for improved robustness.
*   **Pre-built Tasks:** A wide variety of pre-defined manipulation tasks.
*   **Policy Baselines:** Support for DP, ACT, DP3, RDT, PI0, and more!

### **Latest Updates:**

*   **2025/07/09**: Updated endpose control mode. See [RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html).
*   **2025/07/08**: Added [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch.
*   **2025/07/02**: Fixed Piper Wrist Bug, please redownload the embodiment asset.
*   **2025/07/01**: Released the Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)].
*   **2025/06/21**: Released RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)].
*   **2025/04/11**: RoboTwin is selected as a *CVPR Highlight paper*!
*   **2025/02/27**: RoboTwin is accepted to *CVPR 2025*!
*   **2024/09/30**: RoboTwin (Early Version) received *the Best Paper Award at the ECCV Workshop*!
*   **2024/09/20**: Officially released RoboTwin.

### Overview
*   **2.0 Version Branch**: [main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (latest)
*   **1.0 Version Branch**: [1.0 Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0)
*   **1.0 Version Code Generation Branch**: [1.0 Version GPT](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt)
*   **Early Version Branch**: [Early Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version)
*   **第十九届“挑战杯”人工智能专项赛分支**: [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025)
*   **CVPR 2025 Challenge Round 1 Branch**: [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1)
*   **CVPR 2025 Challenge Round 2 Branch**: [CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2)


### Installation

Follow the detailed instructions in the [RoboTwin 2.0 Documentation (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for a 20-minute installation process.

### Tasks Information

Explore the range of tasks in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

### Usage

See [RoboTwin 2.0 Documentation (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for in-depth details.

#### Data Collection

We provide [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) containing over 100,000 pre-collected trajectories. However, it is recommended to collect data yourself for greater control.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

#### Task Running and Data Collection

Run the following command to collect data for a specific task:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

#### Task Configuration

Find detailed configuration options in the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

### Policy Baselines

#### Policies Supported

[DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)

[TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)

Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, DexVLA, OpenVLA-OFT, SmolVLA, AVR, UniVLA

### Experiment & LeaderBoard

We recommend exploring the following topics using the RoboTwin Platform:

*   Single-task fine-tuning capability
*   Visual robustness
*   Language diversity robustness (language condition)
*   Multi-task capability
*   Cross-embodiment performance

Coming Soon.

### Citations

If you find our work helpful, please cite:

**RoboTwin 2.0**:
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin**:
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

**RoboTwin - Benchmarking Generalizable Bimanual Manipulation:**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (Early Version)**:
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

### Acknowledgements

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos

Code Style: `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

### Get Started!

Explore the capabilities of RoboTwin and contribute to the future of robotic manipulation!  For questions or suggestions, contact [Tianxing Chen](https://tianxingchen.github.io).

### License

This project is licensed under the [MIT License](LICENSE).

[Back to top](#)
```
Key improvements and explanations:

*   **SEO-Optimized Title & Introduction:** Changed the title to be more descriptive and include the main keyword ("Bimanual Robotic Manipulation") and a strong hook to grab attention immediately. The first paragraph quickly summarizes what the project is.
*   **Clear Structure:**  Uses headings and subheadings for better organization.
*   **Key Features as Bullet Points:** Makes the core benefits and functionalities easily scannable.
*   **Concise Descriptions:** Keeps descriptions brief and to the point.
*   **Calls to Action:**  Encourages readers to explore the platform.
*   **Updated Information:**  Includes the latest updates from the original README.
*   **Links:** Provides links to key resources (webpage, documentation, paper, original repo, etc.) for easy access.
*   **Clear Installation Instructions:**  Highlights the installation process.
*   **Task Information:**  Clearly outlines tasks.
*   **Code Snippets:**  Includes a code snippet to show how to run data collection (with explanation).
*   **Citations Section:** Includes formatting for citations.
*   **Acknowledgement & Contact:** Keeps these sections.
*   **MIT License:** Includes the correct license tag.
*   **Back to Top Link:**  Provides an internal link for easy navigation.
*   **Concise and clear:** The content has been trimmed down to be more direct and easier to understand.
*   **Bolded crucial parts:** Key phrases are now bolded for emphasis