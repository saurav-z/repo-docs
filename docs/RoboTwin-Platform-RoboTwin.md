# RoboTwin: Unleashing the Power of Bimanual Robotic Manipulation

**[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) is a comprehensive platform for benchmarking and advancing bimanual robotic manipulation, featuring a scalable data generator, strong domain randomization, and a suite of pre-trained policy baselines.**

## Key Features:

*   **Scalable Data Generation:** Generate diverse and realistic data for training and evaluating bimanual robotic manipulation models.
*   **Strong Domain Randomization:** Enhance the robustness of your models through advanced domain randomization techniques.
*   **Pre-trained Policy Baselines:** Leverage readily available baselines including DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA to jumpstart your research.
*   **Diverse Tasks:** Explore a wide range of bimanual manipulation tasks, from simple object manipulation to complex collaborative actions (50+ tasks supported).
*   **Leaderboard & Evaluation:** Evaluate and compare your models on our comprehensive leaderboard, fostering a collaborative research environment.
*   **Active Community:** Join our community and contribute to the advancement of bimanual robotic manipulation research.

## What's New:

*   **RoboTwin 2.0:**  Includes a significant update with enhanced data generation and benchmark capabilities. [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:**  Technical Report available. [PDF](https://arxiv.org/pdf/2506.23351)

## Quick Start:

### Installation:

Follow the detailed installation instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). It typically takes about 20 minutes.

### Tasks and Usage:

1.  **Tasks Information:**  Refer to the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for detailed information on supported tasks.
    <p align="center">
      <img src="./assets/files/50_tasks.gif" width="100%">
    </p>

2.  **Usage - Data Collection:** Follow the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for data collection.

    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

3.  **Modify Task Config:** See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

### Policy Baselines:

We provide a list of already supported policies: [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html), [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html), [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html).

### Deploy Your Policy:
Deploy your policy with the [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

### Pre-collected Large-scale Dataset:

Explore the [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Leaderboard:

Visit the leaderboard at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard) to view performance results and explore experiment settings.

## Citations:

Please cite our work if you find it useful:

**RoboTwin 2.0:**
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin (CVPR 2025 Highlight):**
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

**Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop**
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (Early Version - ECCV Best Paper Award):**
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

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.