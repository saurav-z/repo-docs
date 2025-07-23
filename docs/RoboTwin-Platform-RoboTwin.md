# RoboTwin: Revolutionizing Bimanual Robotic Manipulation (Latest Version 2.0)

**RoboTwin** is a cutting-edge bimanual robotic manipulation platform designed to advance the field of robotics through comprehensive benchmarking, scalable data generation, and strong domain randomization. [Explore the RoboTwin platform on GitHub](https://github.com/RoboTwin-Platform/RoboTwin).

**Key Features:**

*   **Scalable Data Generator:** Easily generate large datasets for training and evaluating bimanual manipulation models.
*   **Robust Domain Randomization:** Implements strong domain randomization techniques to enhance the generalization capabilities of robotic systems.
*   **Comprehensive Benchmark:** Provides a standardized benchmark for evaluating the performance of bimanual robotic manipulation algorithms.
*   **Diverse Tasks:** Supports a wide range of bimanual manipulation tasks, enabling thorough performance assessment.
*   **Open-Source:** Freely available under the MIT license, fostering collaboration and innovation.

**Latest Developments:**

*   **RoboTwin 2.0:** (Under Review 2025) -  Features a significant upgrade in data generation capabilities and robustness. [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR '25 MEIS Workshop:** Official Technical Report available [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)
*   **CVPR 2025 Highlight Paper:** RoboTwin 1.0 accepted to CVPR 2025. [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **ECCV Workshop 2024 Best Paper Award:** Early version of RoboTwin. [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

**Quick Links:**

*   [Webpage](https://robotwin-platform.github.io/)
*   [Document](https://robotwin-platform.github.io/doc)
*   [Paper](https://arxiv.org/abs/2506.18088)
*   [Community](https://robotwin-platform.github.io/doc/community/index.html)

**Installation:**

Detailed installation instructions are available in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).

**Tasks:**

RoboTwin 2.0 features diverse manipulation tasks. See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.
<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

**Usage:**

1.  **Data Collection:**  Collect your own data by following the instructions in the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).
2.  **Task Running:** Execute tasks with a simple command:
    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```
3.  **Task Configuration:** Customize tasks using the configurations described in the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

**Policy Baselines:**

*   DP, ACT, DP3, RDT, PI0, TinyVLA, DexVLA.

**Citations:**

If you use RoboTwin, please cite the following papers:

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

**License:**

This project is released under the [MIT License](LICENSE).

**Contact:**

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).