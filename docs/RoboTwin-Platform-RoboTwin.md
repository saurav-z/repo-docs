# RoboTwin: A Cutting-Edge Bimanual Robotic Manipulation Platform

**RoboTwin** is a state-of-the-art platform designed to advance research in bimanual robotic manipulation, offering a scalable data generator and benchmark for robust performance. Explore the platform, contribute to the community, and view the leaderboard at the [original RoboTwin repo](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Scalable Data Generation:** RoboTwin 2.0 provides a robust and scalable environment for generating diverse and realistic bimanual manipulation data.
*   **Strong Domain Randomization:**  Achieve robust performance through advanced domain randomization techniques, enhancing generalization capabilities.
*   **Comprehensive Benchmark:**  Evaluate and compare the performance of your algorithms on a wide range of bimanual manipulation tasks.
*   **Open-Source and Accessible:**  Leverage pre-collected datasets, and explore a variety of baselines, deploy your policy, and contribute to the advancement of robotic manipulation.
*   **Active Community:** Join the community and collaborate with researchers, access documentation, and view the RoboTwin [Leaderboard](https://robotwin-platform.github.io/leaderboard) to compare your models.
*   **Cutting-Edge Research:**  Based on the latest research and published in top-tier conferences:
    *   **RoboTwin 2.0**: <i>Under Review 2025</i> - [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088)
    *   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR '25 MEIS Workshop** : [PDF](https://arxiv.org/pdf/2506.23351)
    *   **RoboTwin 1.0**: <i>CVPR 2025 (Highlight)</i> - [PDF](https://arxiv.org/pdf/2504.13059)
    *   **RoboTwin Early Version**: <i>ECCV Workshop 2024 (Best Paper Award)</i> - [PDF](https://arxiv.org/pdf/2409.02920)

## Updates:

*   **2025/08/06:** RoboTwin 2.0 Leaderboard Released! [leaderboard website](https://robotwin-platform.github.io/leaderboard)
*   **2025/07/23:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/01:** Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop released [[arXiv](https://arxiv.org/abs/2506.23351)] !
*   **2025/06/21:** RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] is released!
*   **2025/04/11:** RoboTwin selected as a CVPR Highlight paper!
*   **2025/02/27:** RoboTwin accepted to CVPR 2025!
*   **2024/09/30:** RoboTwin (Early Version) received the Best Paper Award at the ECCV Workshop!
*   **2024/09/20:** RoboTwin officially released.

## Installation:

Follow the instructions in the RoboTwin 2.0 Document (Usage - Install & Download)[https://robotwin-platform.github.io/doc/usage/robotwin-install.html] to install the platform (approximately 20 minutes).

## Tasks and Usage:

Explore the diverse tasks and usage details at RoboTwin 2.0 Tasks Doc [https://robotwin-platform.github.io/doc/tasks/index.html].

![RoboTwin Tasks](https://github.com/RoboTwin-Platform/RoboTwin/raw/main/assets/files/50_tasks.gif)

*   **Data Collection:**  Easily collect your own data using the provided scripts.
    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

*   **Task Configuration:** Customize tasks using the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines:

RoboTwin supports the following policy baselines:

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html)
*   [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
*   [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
*   [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
*   [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
*   [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)
*   [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) (Contributed by IRPN Lab, HKUST(GZ))

Deploy your policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## Experiments & Leaderboard:

RoboTwin enables you to investigate:

*   Single-task fine-tuning.
*   Visual robustness.
*   Language diversity robustness.
*   Multi-task capability.
*   Cross-embodiment performance.

Find the full leaderboard and settings at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Dataset:

Access the pre-collected large-scale dataset on [Hugging Face](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations:

If you use RoboTwin, please cite the relevant publications:

```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

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

```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements:

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

## License:

RoboTwin is released under the MIT license. See [LICENSE](./LICENSE) for more details.

## Contact:

For questions or suggestions, contact [Tianxing Chen](https://tianxingchen.github.io).