# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

**RoboTwin** provides a comprehensive platform for research in bimanual robotic manipulation, featuring a scalable data generator, strong domain randomization, and diverse tasks. [Explore the RoboTwin platform on GitHub](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Easily generate large, diverse datasets for training and evaluating robotic manipulation models.
*   **Strong Domain Randomization:** Robustly trains models that generalize well to real-world environments.
*   **Diverse Task Suite:** Includes a wide range of bimanual manipulation tasks, allowing for comprehensive performance evaluation.
*   **Benchmark & Leaderboard:** [Coming Soon] Track progress with experiment results and leaderboards.
*   **Open-Source:**  Released under the MIT license for easy use and modification.

## What's New

*   **RoboTwin 2.0** (Latest Version): Includes new features and improvements for enhanced performance and usability.
    *   [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Community](https://robotwin-platform.github.io/doc/community/index.html)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR '25 MEIS Workshop**
    *   Technical Report: [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0:** The original benchmark, accepted to CVPR 2025 (Highlight).
    *   [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **Early Version:**  Awarded the Best Paper Award at the ECCV Workshop 2024.
    *   [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Quick Start

### Installation

Follow the detailed installation instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).  Installation typically takes around 20 minutes.

### Usage

1.  **Task Information:** See the [RoboTwin 2.0 Tasks Documentation](https://robotwin-platform.github.io/doc/tasks/index.html) for a complete list of available tasks.
    <p align="center">
      <img src="./assets/files/50_tasks.gif" width="100%">
    </p>
2.  **Data Collection:**  We provide pre-collected data, but recommend generating your own for best results.
    *   See [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
    *   Run a task using:
        ```bash
        bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
        # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
        ```
        <img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">
3.  **Task Configuration:**  Customize task parameters using the configurations detailed in the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

### Policy Baselines

Explore available policies: [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html), [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html), and deploy your own using the [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## Roadmap

*   [ ]  G3Flow
*   [ ]  HybridVLA
*   [ ]  DexVLA
*   [ ]  OpenVLA-OFT
*   [ ]  SmolVLA
*   [ ]  AVR
*   [ ]  UniVLA

## Citations

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

**RoboTwin Challenge:**

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

## Acknowledgements

**Software Support:** D-Robotics, **Hardware Support:** AgileX Robotics, **AIGC Support:** Deemos

## License

This project is released under the MIT License.  See [LICENSE](./LICENSE) for details.

## Contact

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).