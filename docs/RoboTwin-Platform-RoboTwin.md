# RoboTwin: Your Gateway to Cutting-Edge Bimanual Robotic Manipulation

RoboTwin is a comprehensive platform for research in bimanual robotic manipulation, offering a scalable data generator, robust domain randomization, and a benchmark for evaluating and comparing algorithms.  Explore the platform on GitHub: [RoboTwin Repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Scalable Data Generation:** Generate massive datasets to train and evaluate your bimanual robotic manipulation algorithms.
*   **Strong Domain Randomization:** Enhance the robustness and generalization capabilities of your models through advanced domain randomization techniques.
*   **Comprehensive Benchmark:** Evaluate your models against a diverse set of bimanual robotic manipulation tasks.
*   **Pre-collected Datasets:** Access over 100,000 pre-collected trajectories in the [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
*   **Policy Baselines:**  Get started quickly with support for multiple policy baselines including DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA, and LLaVA-VLA.
*   **Active Development:** Regularly updated with new features, tasks, and improvements.

## What's New:

*   **RoboTwin 2.0 Release:** The latest version with enhanced features, improved performance, and a dedicated [Leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **CVPR 2025 Recognition:** RoboTwin selected as a CVPR Highlight paper and accepted to CVPR 2025.
*   **ECCV Workshop Best Paper:** Early version of RoboTwin received the Best Paper Award at the ECCV Workshop 2024.

## Get Started:

### Installation:

Follow the detailed installation instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) (installation time is approximately 20 minutes).

### Usage:

*   **Tasks Information:**  Explore the available tasks via the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).
*   **Data Collection:**  Collect your own data using the provided scripts.
    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```
*   **Configuration:**  Customize tasks using the configurations detailed in the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).
*   **Policy Deployment:** Deploy and evaluate your own policies using the guide at [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## Documentation:

*   **RoboTwin 2.0 Document (Usage):** [https://robotwin-platform.github.io/doc/usage/index.html](https://robotwin-platform.github.io/doc/usage/index.html)

## Leaderboard:

*   **RoboTwin Leaderboard:** [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard)

## Citations:

Please cite the following papers if you use this work:

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

## Acknowledgements:

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

## Contact:

For questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License:

This project is released under the MIT License. See [LICENSE](./LICENSE) for more details.