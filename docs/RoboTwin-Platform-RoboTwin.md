# RoboTwin: The Ultimate Bimanual Robotic Manipulation Platform

[**RoboTwin**](https://github.com/RoboTwin-Platform/RoboTwin) is a cutting-edge, scalable platform and benchmark designed to advance the field of bimanual robotic manipulation.

## Key Features:

*   **Scalable Data Generation:**  Generates massive datasets with strong domain randomization for robust training.
*   **Comprehensive Benchmark:** Provides a standardized platform for evaluating and comparing bimanual robotic manipulation algorithms.
*   **Diverse Tasks:** Supports a wide range of bimanual manipulation tasks to test the capabilities of different approaches.
*   **Pre-collected Datasets:** Access over 100,000 pre-collected trajectories to kickstart your research.
*   **Policy Baselines:** Offers implementations of various state-of-the-art policies for easy experimentation.
*   **Active Community & Leaderboard:** Track your progress and compare your performance on the public leaderboard.

## Key Resources:

*   **Latest Version (RoboTwin 2.0):**
    *   [Webpage](https://robotwin-platform.github.io/)
    *   [Documentation](https://robotwin-platform.github.io/doc/)
    *   [Paper (arXiv)](https://arxiv.org/abs/2506.18088)
    *   [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:**
    *   [Technical Report (arXiv)](https://arxiv.org/abs/2506.23351)

## 🚀 What's New in RoboTwin 2.0

*   **Released RoboTwin 2.0 Leaderboard** [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **ChinaSI 2025** Received Outstanding Poster at (Ranking 1st).
*   **Fix DP3 evaluation code error**, Will update RoboTwin 2.0 paper next week.
*   **Update endpose control mode**, See [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)] for more details.
*   **Challenge-Cup-2025** Branch is available ([Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) ).
*   **Piper Wrist Bug** Fixed [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)].
*   **RoboTwin 2.0 Released** [[Webpage](https://robotwin-platform.github.io/)] !
*   **CVPR Highlight paper!**
*   **Accepted to *CVPR 2025* !**
*   **ECCV Workshop 2024 (Best Paper Award) Early Version**

## ⚙️ Installation

Follow the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions, which typically takes around 20 minutes.

## 🤖 Tasks Information

Explore the diverse range of tasks supported by RoboTwin in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## 🧑‍💻 Usage

### Document

For detailed guidance, consult the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

While pre-collected data is available on [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset), users are encouraged to collect their own data for optimal results due to the configurability of the platform.

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

To run a task and collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Modify Task Config

Customize task configurations by referring to the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## 💡 Policy Baselines

### Policies Supported:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   OpenVLA-oft
*   TinyVLA
*   DexVLA
*   LLaVA-VLA

### Deploy Your Policy:

See [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

⏰ TODO: G3Flow, HybridVLA, SmolVLA, AVR, UniVLA

## 📊 Experiment & LeaderBoard

The RoboTwin Platform is ideal for exploring:

*   Single-task fine-tuning
*   Visual robustness
*   Language diversity robustness (language condition)
*   Multi-task capabilities
*   Cross-embodiment performance

Find the full leaderboard at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## 💾 Pre-collected Large-scale Dataset

Access the extensive pre-collected dataset on [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## 🙏 Citations

If you use RoboTwin in your research, please cite the following:

```bibtex
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

```bibtex
@InProceedings{Mu_2025_CVPR,
    author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
    title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27649-27660}
}
```

```bibtex
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

```bibtex
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## 💖 Acknowledgements

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

## ✉️ Contact

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## 📜 License

This project is released under the MIT license. See the [LICENSE](./LICENSE) file for details.