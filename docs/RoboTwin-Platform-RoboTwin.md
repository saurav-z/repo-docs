# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) is a comprehensive platform designed to advance the field of bimanual robotic manipulation through robust benchmarking, scalable data generation, and strong domain randomization.

## Key Features:

*   **Scalable Data Generation:** Create diverse datasets for training and evaluating bimanual robotic manipulation algorithms.
*   **Strong Domain Randomization:** Enhance the robustness of your models with realistic simulation environments.
*   **Multiple Versions:** Explore different versions of RoboTwin, including the latest **RoboTwin 2.0** and early versions.
*   **Open-Source and Community-Driven:** Benefit from a community of researchers and developers.
*   **Benchmark and Challenge:** Access official technical reports and participate in challenges like the RoboTwin Dual-Arm Collaboration Challenge at CVPR'25 MEIS Workshop.
*   **Comprehensive Documentation:**  Explore in-depth documentation and user guides: [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Community](https://robotwin-platform.github.io/doc/community/index.html)
*   **Policy Support:** Includes a variety of supported policy baselines: DP, ACT, DP3, RDT, PI0, TinyVLA, DexVLA, and LLaVA-VLA

## What's New in RoboTwin 2.0?

RoboTwin 2.0 offers significant improvements:

*   **Enhanced Data Generation:** Create more diverse and realistic datasets with advanced simulation capabilities.
*   **Improved Domain Randomization:** Achieve better generalization and robustness in real-world scenarios.
*   **Expanded Benchmarking Capabilities:** Evaluate your models on a wider range of tasks and scenarios.

## Quick Start:

1.  **Installation:** Follow the detailed installation instructions:  [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) (approximately 20 minutes).
2.  **Task Information:** Explore the available tasks and details: [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html)
3.  **Data Collection:** Start by running the data collection script:

    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

4.  **Configuration:** Customize task configurations.  See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html)

## Updates:

*   **2025/07/23:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025.
*   **2025/07/19:** Fixed DP3 evaluation code error.
*   **2025/07/09:** Updated endpose control mode.
*   **2025/07/08:** Added "挑战杯" branch.
*   **2025/07/02:** Fixed Piper Wrist Bug.
*   **2025/07/01:** Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop.
*   **2025/06/21:** Released RoboTwin 2.0.
*   **2025/04/11:** RoboTwin selected as CVPR Highlight paper!
*   **2025/02/27:** RoboTwin is accepted to CVPR 2025 !
*   **2024/09/30:** RoboTwin (Early Version) received the Best Paper Award at the ECCV Workshop!
*   **2024/09/20:** Officially released RoboTwin.

## Citations:

If you use RoboTwin in your research, please cite the following papers:

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

**RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop:**

```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (early version):**

```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements:

*   **Software Support:** D-Robotics
*   **Hardware Support:** AgileX Robotics
*   **AIGC Support:** Deemos

## License:

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact:

For questions and suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).