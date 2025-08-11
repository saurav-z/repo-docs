# RoboTwin: The Premier Bimanual Robotic Manipulation Platform

**Unleash the power of dual-arm robotics with RoboTwin, a cutting-edge platform for research and development in bimanual manipulation.** [Explore the RoboTwin Repository](https://github.com/RoboTwin-Platform/RoboTwin)

## Key Features

*   **Scalable Data Generation:** RoboTwin 2.0 offers a robust data generation pipeline.
*   **Strong Domain Randomization:**  Enhance the generalization capabilities of your models.
*   **Comprehensive Benchmark:**  Evaluate and compare bimanual robotic manipulation algorithms.
*   **Pre-collected Datasets:** Access over 100,000 pre-collected trajectories for efficient research.
*   **Multiple Policy Baselines:**  Explore state-of-the-art policy baselines including DP, ACT, DP3, RDT, PI0, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Open-Source and Accessible:** RoboTwin is released under the MIT license for open use.
*   **Community & Collaboration:**  Connect with researchers through the community resources.

## What's New in RoboTwin 2.0

*   **RoboTwin 2.0 Leaderboard:**  Compete and compare your results at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **CVPR 2025 Recognition:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 and RoboTwin was selected as a CVPR Highlight paper!
*   **Latest Updates:** Regularly updated with new features, bug fixes, and improvements.

## Getting Started

### Installation

1.  Follow the detailed installation instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes around 20 minutes.

### Tasks and Usage

*   Explore various tasks available in [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).
*   For more details on usage, please refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

*   Easily collect data by running the following command, replacing the placeholders with your desired task and configuration:

    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

### Configurations

*   Refer to [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for task configurations.

## Explore RoboTwin

*   **Webpage:** [https://robotwin-platform.github.io/](https://robotwin-platform.github.io/)
*   **Documentation:** [https://robotwin-platform.github.io/doc/](https://robotwin-platform.github.io/doc/)
*   **Leaderboard:** [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **Paper:** [https://arxiv.org/abs/2506.18088](https://arxiv.org/abs/2506.18088)
*   **Community:** [https://robotwin-platform.github.io/doc/community/index.html](https://robotwin-platform.github.io/doc/community/index.html)

## Citations

Please cite the following papers if you use RoboTwin in your research:

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

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos

## Contact

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).