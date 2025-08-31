# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

**RoboTwin** offers a comprehensive platform for researchers to benchmark and advance the field of bimanual robotic manipulation. Check out the original repository [here](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Scalable and Realistic:** RoboTwin 2.0 provides a scalable data generator with strong domain randomization to improve robustness.
*   **Comprehensive Benchmarking:** Evaluate your algorithms with a suite of challenging tasks and a detailed leaderboard.
*   **Extensive Documentation:** Access detailed documentation, including installation guides, task specifications, and baseline policy implementations.
*   **Large-Scale Dataset:** Utilize a pre-collected, large-scale dataset of over 100,000 trajectories to jumpstart your research.
*   **Community and Collaboration:** Join a vibrant community to discuss research, share insights, and contribute to the advancement of bimanual robotics.

## RoboTwin Versions & Achievements

*   **RoboTwin 2.0 (Latest)**:  [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
    *   Outstanding Poster at ChinaSI 2025 (Ranking 1st).
    *   Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
    *   Highlighted in CVPR 2025
    *   Released June 21, 2025
*   **RoboTwin 1.0**: 
    *   Accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>: [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **RoboTwin (Early Version)**: 
    *   Awarded <i>the Best Paper Award  at the ECCV Workshop</i>: [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Quick Start:

### Installation

See [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for installation instructions. Installation typically takes about 20 minutes.

### Tasks

Explore a range of tasks. See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

### Usage

*   **Document:**  Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for in-depth guidance.

*   **Data Collection:** Use provided scripts to collect data.  We strongly recommend collecting your own data for optimal results.

    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

*   **Task Configuration:**  Customize tasks. See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for details.

## Policy Baselines

*   **Supported Policies:** DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, LLaVA-VLA
*   **Deploy Your Policy:** [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

Explore the performance of RoboTwin on various topics, with a focus on single-task tuning, visual robustness, language diversity, multi-task capabilities, and cross-embodiment performance.  Check the full leaderboard at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Dataset

Access a pre-collected large-scale dataset [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you find our work useful, please consider citing:

**RoboTwin 2.0**
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin**
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
**RoboTwin Dual-Arm Collaboration Challenge**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```
**RoboTwin (Early Version)**
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

## Contact

Contact [Tianxing Chen](https://tianxingchen.github.io) for questions or suggestions.

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.