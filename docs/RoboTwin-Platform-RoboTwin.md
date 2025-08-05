# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

RoboTwin is a cutting-edge platform designed to advance the field of bimanual robotic manipulation, offering a scalable data generator, a robust benchmark, and a vibrant community.  Explore the possibilities and contribute to the future of robotics!  [Visit the original RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Scalable Data Generation:**  Create diverse and realistic datasets for training and evaluating your robotic manipulation algorithms.
*   **Strong Domain Randomization:**  Enhance the robustness of your models through comprehensive domain randomization techniques.
*   **Comprehensive Benchmarking:**  Evaluate your algorithms across a wide range of bimanual manipulation tasks.
*   **Active Community:** Engage with a community of researchers and developers, and collaborate on the future of robotics.
*   **Multiple Versions and Challenges:** Including RoboTwin 2.0, RoboTwin 1.0, and participation in the CVPR'25 MEIS Workshop challenge.

## Recent Updates & Highlights:

*   **RoboTwin 2.0:** The latest version includes enhancements to data generation, domain randomization, and task variety.
    *   Released July 2025
    *   Received Outstanding Poster at ChinaSI 2025 (Ranking 1st)
    *   Endpose Control Mode
*   **CVPR 2025 Recognition:**  RoboTwin has been recognized as a CVPR Highlight paper and features in the CVPR 2025 MEIS Workshop challenge.
*   **ECCV 2024 Best Paper Award:** The early version of RoboTwin received the Best Paper Award at the ECCV Workshop 2024.

## Getting Started

### Installation

Follow the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for installation instructions.  The installation process typically takes around 20 minutes.

### Tasks & Usage

RoboTwin offers a rich set of tasks for bimanual manipulation.  Details can be found in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

**Data Collection:**  While pre-collected trajectories are available in the [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset), we recommend custom data collection to leverage the configurability of tasks and embodiments.

**Running Tasks & Data Collection:**

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

**Task Configuration:**  See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for detailed task configuration options.

### Policy Baselines

RoboTwin supports various policy baselines, including:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   TinyVLA
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))

### Experiment & Leaderboard

Explore these topics to advance bimanual manipulation:

1.  Single-task fine-tuning capabilities
2.  Visual robustness
3.  Language diversity robustness (language condition)
4.  Multi-task capabilities
5.  Cross-embodiment performance
    **Coming Soon**

## Citations

If you find RoboTwin useful, please cite the following papers:

**RoboTwin 2.0:**
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin - CVPR 2025**
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

**RoboTwin - ECCV Workshop 2024**
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgments

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.