# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation

RoboTwin is a cutting-edge platform for developing and evaluating bimanual robotic manipulation skills, offering a comprehensive suite of tools, tasks, and baselines. **[Explore the RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin)** for a deep dive into the world of dual-arm robotics!

## Key Features

*   **Comprehensive Benchmark:** Includes a diverse set of tasks, from simple pick-and-place to complex collaborative manipulations, designed to challenge and evaluate robotic systems.
*   **Advanced Domain Randomization:** Provides robust domain randomization techniques to enhance the generalization capabilities of robotic agents.
*   **Scalable Data Generation:** Offers tools for generating large-scale, diverse datasets to train and evaluate models effectively.
*   **Multiple Baselines:** Supports a range of policy baselines, including DP, ACT, DP3, RDT, PI0, and more.
*   **Open-Source and Community-Driven:** Built for collaboration and open-source contributions, fostering innovation in the field.

## Recent Updates

*   **July 2025:** RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025!
*   **July 2025:** Fixed DP3 evaluation code error.
*   **July 2025:** Updated endpose control mode in documentation.
*   **July 2025:** Added the 第十九届挑战杯分支.
*   **July 2025:** Fixed Piper Wrist Bug, please redownload the embodiment asset.
*   **July 2025:** RoboTwin Dual-Arm Collaboration Challenge Technical Report released on arXiv!
*   **June 2025:** RoboTwin 2.0 released, introducing significant advancements!
*   **April 2025:** RoboTwin was selected as a CVPR Highlight paper!
*   **February 2025:** RoboTwin accepted to CVPR 2025!
*   **September 2024:** RoboTwin (Early Version) received the Best Paper Award at the ECCV Workshop!
*   **September 2024:** Officially released RoboTwin.

## Getting Started

### Installation

Detailed installation instructions are available in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html), which should take approximately 20 minutes to complete.

### Tasks Overview

Explore the various tasks supported by RoboTwin in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

### Usage

For detailed usage instructions, please refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

#### Data Collection

We strongly recommend users to perform data collection themselves to leverage the configurability and diversity of RoboTwin.

<img src="./assets/files/domain_randomization.png" alt="Domain Randomization" style="display: block; margin: auto; width: 100%;">

#### 1. Task Running and Data Collection

Run the command below to collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

#### 2. Task Configuration

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

### Policy Baselines

Explore the implemented policy baselines: [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group).

Deploy your own policies following the [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## Citation

If you find this work useful, please consider citing:

**RoboTwin 2.0**: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin**: Dual-Arm Robot Benchmark with Generative Digital Twins, accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>
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

Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin**: Dual-Arm Robot Benchmark with Generative Digital Twins (early version), accepted to <i style="color: red; display: inline;"><b>ECCV Workshop 2024 (Best Paper Award)</b></i>
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

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.

## Contact

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).