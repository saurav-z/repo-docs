# RoboTwin: A Comprehensive Benchmark for Bimanual Robotic Manipulation

**RoboTwin** is a cutting-edge platform designed to advance the field of bimanual robotic manipulation, offering a robust benchmark and a rich set of tools for researchers and developers. Explore the [original repository](https://github.com/RoboTwin-Platform/RoboTwin) for further details.

## Key Features

*   **Scalable Data Generation:** Generate vast amounts of diverse data to train and evaluate robotic manipulation algorithms.
*   **Strong Domain Randomization:** Robustly simulate real-world scenarios to enhance the generalizability of your models.
*   **Comprehensive Benchmark:** Evaluate and compare your algorithms using a standardized set of tasks and metrics.
*   **Multiple Versions:** Access to the latest RoboTwin 2.0 and previous versions, including early releases and challenge-specific branches.
*   **Leaderboard:** Track your progress and compare your results with other researchers on the RoboTwin leaderboard.
*   **Policy Baselines:** Support for several baselines, and guidelines for deploying your own policies.
*   **Large-Scale Dataset:** Access pre-collected data through Hugging Face.
*   **Active Community:** Stay up-to-date with the latest developments through the community resources.

## Versions & Resources

*   **RoboTwin 2.0 (Latest):** A scalable data generator and benchmark with strong domain randomization.
    *   [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop:**
    *   [Technical Report](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0:** Dual-Arm Robot Benchmark with Generative Digital Twins.
    *   *CVPR 2025 Highlight Paper* - [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **RoboTwin (Early Version):**
    *   *ECCV Workshop 2024 Best Paper Award* - [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Updates

*   **(2025/08/28)**: RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088) updated.
*   **(2025/08/25)**: ACT deployment code fixed; [leaderboard](https://robotwin-platform.github.io/leaderboard) updated.
*   **(2025/08/06)**: RoboTwin 2.0 Leaderboard released: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
*   **(2025/07/23)**: RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **(2025/07/19)**: Fixed DP3 evaluation code error.
*   **(2025/07/09)**: Endpose control mode updated.
*   **(2025/07/08)**: [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch released.
*   **(2025/07/02)**: Piper Wrist Bug fixed, re-download embodiment asset.
*   **(2025/07/01)**: Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] released.
*   **(2025/06/21)**: RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] released.
*   **(2025/04/11)**: RoboTwin selected as *CVPR Highlight paper*.
*   **(2025/02/27)**: RoboTwin accepted to *CVPR 2025*.
*   **(2024/09/30)**: RoboTwin (Early Version) received the Best Paper Award at the ECCV Workshop.
*   **(2024/09/20)**: RoboTwin officially released.

## Installation

Follow the detailed installation instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes about 20 minutes.

## Tasks Information

Explore the various tasks and their details in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).

## Usage

### Document

Refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for in-depth instructions.

### Data Collection

It is recommended to collect data, but pre-collected trajectories are available at [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

<img src="./assets/files/domain_randomization.png" alt="Domain Randomization" style="display: block; margin: auto; width: 100%;">

### Task Running and Data Collection

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Modify Task Config

See the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines

*   DP ([Documentation](https://robotwin-platform.github.io/doc/usage/DP.html))
*   ACT ([Documentation](https://robotwin-platform.github.io/doc/usage/ACT.html))
*   DP3 ([Documentation](https://robotwin-platform.github.io/doc/usage/DP3.html))
*   RDT ([Documentation](https://robotwin-platform.github.io/doc/usage/RDT.html))
*   PI0 ([Documentation](https://robotwin-platform.github.io/doc/usage/Pi0.html))
*   OpenVLA-oft ([Documentation](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html))
*   TinyVLA ([Documentation](https://robotwin-platform.github.io/doc/usage/TinyVLA.html))
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))

Deploy Your Policy: [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

The leaderboard and experiment settings can be found at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Access the pre-collected dataset at [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you use RoboTwin, please cite the following papers:

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

## Acknowledgement

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos.

For questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT license. See the [LICENSE](./LICENSE) file for details.