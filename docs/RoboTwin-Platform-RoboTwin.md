# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="75%">
</p>

**RoboTwin is a comprehensive, open-source platform designed to advance the field of bimanual robotic manipulation.  Explore the latest advancements and contribute to the future of robotics!**  [Visit the original repo on GitHub](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Robust & Scalable:** RoboTwin 2.0 features strong domain randomization for realistic simulation and robust performance.
*   **Extensive Task Variety:**  Includes a diverse set of 50 bimanual manipulation tasks, ideal for evaluating and comparing robotic control algorithms.
*   **Open-Source & Accessible:** Free to use, modify, and contribute to, with detailed documentation and community support.
*   **Comprehensive Benchmarks:**  Features leaderboards and standardized evaluation metrics to track progress and compare different approaches.
*   **Policy Baselines:** Includes implementations of popular algorithms like DP, ACT, DP3, RDT, PI0 and more. Supports deploying your own policies.
*   **Active Development & Community:**  Regular updates, including new features, tasks, and policies, and strong community support.
*   **Multiple Versions & Challenges:** Access to Early Version, 1.0 Version, and 2.0 Version along with CVPR challenge datasets and workshops.

## What's New in RoboTwin 2.0

*   **Released Leaderboard:** Explore and compare your results on the [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **Outstanding Poster Award:** Awarded Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **Endpose Control Mode:** Added support for endpose control, providing more control options.
*   **RoboTwin Dual-Arm Collaboration Challenge:** Technical Report available at [arXiv](https://arxiv.org/abs/2506.23351).
*   **CVPR Highlight Paper:** RoboTwin 1.0 was selected as a CVPR Highlight paper.
*   **ECCV Best Paper Award:** RoboTwin (Early Version) received the Best Paper Award at the ECCV Workshop 2024.

## Getting Started

### Installation

Follow the detailed installation instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).  Installation takes approximately 20 minutes.

### Usage

RoboTwin is designed to be easy to use, even for newcomers.

*   **Data Collection:** Collect data yourself to fully utilize the high configurability and diversity of RoboTwin. See [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) for pre-collected trajectories.
*   **Task Running:**  Run tasks and collect data using the provided `collect_data.sh` script.

    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

*   **Task Configuration:**  Customize tasks with different configurations. See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

### Policy Baselines

RoboTwin supports a variety of state-of-the-art policy baselines, including:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   TinyVLA
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))

Learn how to deploy your own policies: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Explore & Contribute

RoboTwin is an excellent platform to explore:

*   Single-task fine-tuning capabilities.
*   Visual robustness in robotic manipulation.
*   Language diversity robustness (language conditioning).
*   Multi-task learning capabilities.
*   Cross-embodiment performance.

## Citations

If you find RoboTwin useful, please consider citing the following:

**RoboTwin 2.0:**
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin 1.0 (CVPR 2025 Highlight):**
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

**RoboTwin Dual-Arm Collaboration Challenge (CVPR 2025 MEIS Workshop):**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (Early Version) (ECCV Workshop Best Paper):**
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

Code Style: `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

For questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.