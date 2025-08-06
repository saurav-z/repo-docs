# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) offers a comprehensive and versatile platform for research in dual-arm robotic manipulation, featuring a scalable data generator, strong domain randomization, and a variety of tasks.

## Key Features:

*   **Scalable Data Generation:** Generate diverse datasets for training and evaluating robotic manipulation models.
*   **Strong Domain Randomization:** Enhance the robustness and generalizability of your models through realistic simulation environments.
*   **Diverse Tasks:** Evaluate your algorithms on a wide range of bimanual manipulation tasks.
*   **Pre-collected Trajectories:** Access a large dataset of pre-collected trajectories to accelerate your research.
*   **Policy Baselines:** Access a variety of policy baselines, including DP, ACT, DP3, RDT, PI0, TinyVLA, and DexVLA.
*   **Community Support:** Benefit from a supportive community through [Webpage](https://robotwin-platform.github.io/), [Document](https://robotwin-platform.github.io/doc), and [Community](https://robotwin-platform.github.io/doc/community/index.html) resources.
*   **Continuous Updates**: Benefit from frequent improvements and updates, including new policy baselines and features.

## What's New:

*   **RoboTwin 2.0** (Latest): A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation. [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** Official Technical Report: [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0:** Accepted to *CVPR 2025 (Highlight)*: [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **Early Version:** Accepted to *ECCV Workshop 2024 (Best Paper Award)*: [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Installation:

Follow the detailed installation instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes about 20 minutes.

## Tasks:

Explore the various bimanual manipulation tasks supported by RoboTwin.  See the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage:

1.  **Task Running and Data Collection:**
    Use the `collect_data.sh` script to run tasks and collect data:

    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

2.  **Task Configuration:**
    Refer to the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for detailed configuration options.

## Policy Baselines:

RoboTwin provides a variety of policy baselines to compare with and test out models:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   TinyVLA
*   DexVLA
*   LLaVA-VLA

## Experiments and Leaderboard:

Coming soon.

## Citations:

If you find our work helpful, please cite the following publications:

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

## Acknowledgements:

**Software Support:** D-Robotics, **Hardware Support:** AgileX Robotics, **AIGC Support:** Deemos

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License:

Released under the MIT License. See [LICENSE](./LICENSE) for details.