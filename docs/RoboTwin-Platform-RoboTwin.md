# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

**RoboTwin** provides a comprehensive platform for evaluating and advancing the field of dual-arm robotic manipulation. Check out the original repo [here](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Scalable Data Generation:** Efficiently generate diverse datasets for training and evaluation.
*   **Strong Domain Randomization:**  Enhances robustness and generalization capabilities.
*   **Bimanual Robotic Manipulation Focus:** Specifically designed for dual-arm robotic tasks.
*   **Comprehensive Benchmarks:**  Includes a leaderboard to compare and track performance.
*   **Multiple Policy Baselines:** Supports diverse policy architectures for experimentation.
*   **Large-scale Dataset:** Provides pre-collected trajectories for immediate use.
*   **Active Community:**  Open-source, with ongoing updates and community engagement.

## What's New:

*   **RoboTwin 2.0:** Latest version with enhanced features and improved performance (Under Review 2025)
    *   [Webpage](https://robotwin-platform.github.io/)
    *   [Document](https://robotwin-platform.github.io/doc)
    *   [Paper](https://arxiv.org/abs/2506.18088)
    *   [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop**
    *   Official Technical Report: [PDF](https://arxiv.org/pdf/2506.23351)
*   **RoboTwin 1.0:**  (CVPR 2025 Highlight)
    *   [PDF](https://arxiv.org/pdf/2504.13059)
*   **RoboTwin (Early Version)**: (ECCV Workshop 2024 Best Paper Award)
    *   [PDF](https://arxiv.org/pdf/2409.02920)

## 🚀 Quick Start:

### Installation

Please refer to the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions. (Approx. 20 minutes)

### Tasks Information

Explore the available tasks in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

### Usage

**Document:**  Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for comprehensive guides.

**Data Collection:**  Collect your own data or leverage pre-collected trajectories.

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

**Modify Task Config:** Consult the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) to customize task settings.

## 🤖 Policy Baselines

RoboTwin supports a variety of policy baselines to get you started:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   OpenVLA-oft
*   TinyVLA
*   DexVLA
*   LLaVA-VLA

**Deploy Your Policy:** Find guidance on [Deploying Your Policy](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## 📊 Experiment & Leaderboard

Explore topics such as single-task fine-tuning, visual robustness, language diversity, multi-task capabilities, and cross-embodiment performance.  The complete leaderboard is available at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## 💾 Pre-collected Large-scale Dataset

Access the pre-collected RoboTwin 2.0 Dataset on Hugging Face: [https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## 📚 Citations

Please cite the relevant papers if you find this work useful:

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

**RoboTwin Challenge:**

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

## 🤝 Acknowledgements

**Software Support:** D-Robotics, **Hardware Support:** AgileX Robotics, **AIGC Support:** Deemos.

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## 📜 License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.