# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

**RoboTwin** offers a cutting-edge platform for advancing bimanual robotic manipulation, providing a scalable data generator, comprehensive benchmark, and robust domain randomization capabilities. Explore the latest advancements at the [RoboTwin GitHub Repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate diverse datasets for training and evaluating bimanual robotic manipulation algorithms.
*   **Comprehensive Benchmark:** Evaluate your algorithms across a wide range of challenging tasks and metrics.
*   **Strong Domain Randomization:** Enhance robustness and generalization through realistic simulation and randomization techniques.
*   **Cutting-Edge Baselines:** Utilize state-of-the-art baselines including DP, ACT, DP3, RDT, and more.
*   **Active Community:** Engage with researchers and developers through the official [RoboTwin Webpage](https://robotwin-platform.github.io/) and [Community](https://robotwin-platform.github.io/doc/community/index.html).
*   **Challenge & Leaderboard:** Participate in the [RoboTwin Dual-Arm Collaboration Challenge@CVPR'25 MEIS Workshop](https://robotwin-platform.github.io/leaderboard) and climb the leaderboard.

## Latest Updates

*   **RoboTwin 2.0 Released!** [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge** at CVPR 2025 MEIS Workshop. [Technical Report](https://arxiv.org/abs/2506.23351)
*   **CVPR 2025 Highlight Paper!** RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins.

## 🚀 Getting Started

### Installation

Follow the detailed installation instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to get started.

### Tasks

Explore the various bimanual manipulation tasks available within RoboTwin. For more details see [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html)

### Usage

1.  **Data Collection:**

    Collect data using the provided scripts. Data collection is highly configurable.

    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

2.  **Task Configuration:**

    Customize task configurations to suit your needs. See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more information.

## 🤖 Policy Baselines

RoboTwin supports a range of policy baselines to accelerate your research:

*   DP, ACT, DP3, RDT, PI0
*   TinyVLA, DexVLA
*   LLaVA-VLA
*   More coming soon!

Deploy your own policies with the [deployment guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## 🏆 Experiment & Leaderboard

Use RoboTwin to explore:

1.  Single-task fine-tuning
2.  Visual robustness
3.  Language diversity robustness
4.  Multi-task capability
5.  Cross-embodiment performance

Find the full leaderboard and experiment settings at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## 📝 Citations

Please cite our work if you find RoboTwin useful:

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

## 🙏 Acknowledgements

Special thanks to: D-Robotics (Software Support), AgileX Robotics (Hardware Support), and Deemos (AIGC Support).

For questions and suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## 📄 License

This project is licensed under the MIT License.  See the [LICENSE](./LICENSE) file for details.