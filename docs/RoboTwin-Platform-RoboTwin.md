# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

**RoboTwin** is a comprehensive platform for researchers and developers to advance the field of bimanual robotic manipulation, providing a scalable data generator and benchmark with strong domain randomization. ([Original Repo](https://github.com/RoboTwin-Platform/RoboTwin))

## Key Features:

*   **Scalable Data Generation:** Generate diverse datasets with ease for robust training.
*   **Strong Domain Randomization:** Achieve significant performance improvements through effective domain randomization techniques.
*   **Comprehensive Benchmark:** Evaluate your algorithms on a variety of challenging bimanual manipulation tasks.
*   **Multiple Versions and Challenges:** Explore different versions (2.0, 1.0, and early versions) and participate in the ongoing RoboTwin Dual-Arm Collaboration Challenge.
*   **Community Support:** Access detailed documentation, a community forum, and pre-collected datasets to accelerate your research.
*   **Policy Baselines:** Evaluate your models using available baselines, including DP, ACT, DP3, RDT, PI0, and more.

## Key Updates:

*   **RoboTwin 2.0:** The latest version, offering significant improvements in data generation and benchmark capabilities (Under Review 2025).  [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088)
*   **CVPR 2025 Challenge:** Participate in the ongoing RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop. [Technical Report](https://arxiv.org/pdf/2506.23351)
*   **CVPR 2025 Highlight Paper:** RoboTwin 1.0 was accepted to CVPR 2025 as a Highlight paper!  [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **ECCV Workshop 2024 Best Paper:** Early version of RoboTwin received the Best Paper Award at the ECCV Workshop 2024!  [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Getting Started

### Installation

Follow the detailed instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to install the platform. Installation typically takes approximately 20 minutes.

### Tasks

Explore the various bimanual manipulation tasks available on the platform, including:

*   Beat Block Hammer
*   ... (many more, see RoboTwin 2.0 Tasks Doc)

For more details, see [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

### Usage

1.  **Data Collection:** Data Collection is designed to be user-friendly, but it is important to note that it is necessary to perform data collection yourself due to the high configurability and diversity of task and embodiment setups.
    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```
2.  **Configuration:** Customize your experiments by modifying task configurations. For more details, see [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines

The platform offers the following baselines. For more details, please refer to:

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html)
*   [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
*   [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
*   [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
*   [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
*   [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html)
*   [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html)

### Deploy Your Policy
[guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Citations

If you utilize RoboTwin in your research, please cite the following:

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

## License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).