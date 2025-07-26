# RoboTwin: The Premier Benchmark for Bimanual Robotic Manipulation 🤖

[View the RoboTwin Repository on GitHub](https://github.com/RoboTwin-Platform/RoboTwin)

RoboTwin is a cutting-edge platform providing a comprehensive benchmark for bimanual robotic manipulation, featuring advanced data generation, strong domain randomization, and a variety of challenging tasks.

## Key Features

*   **Dual-Arm Manipulation:** Evaluate and benchmark algorithms on complex bimanual tasks.
*   **Advanced Domain Randomization:** Robustly test algorithms against real-world variations.
*   **Scalable Data Generation:** Easily generate large datasets for training and evaluation.
*   **Variety of Tasks:** Explore a diverse set of manipulation challenges.
*   **Open-Source and Accessible:** Available under the MIT license for research and development.
*   **Comprehensive Documentation:** Detailed documentation and resources available for getting started.
*   **Multiple Versions:** Access different versions of the platform for different research needs.

## Versions & Recent Updates

*   **RoboTwin 2.0 (Latest):** [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088)
    *   Outstanding Poster at ChinaSI 2025 (Ranking 1st).
    *   Updates to the endpose control mode.
    *   Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)]
    *   Released RoboTwin 2.0
    *   CVPR Highlight paper.
    *   Accepted to CVPR 2025.

*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)

*   **RoboTwin 1.0:** Accepted to CVPR 2025 (Highlight)
    *   [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)

*   **Early Version:** Accepted to ECCV Workshop 2024 (Best Paper Award)
    *   [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Installation

See [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions.

## Tasks and Usage

RoboTwin offers a wide range of tasks to challenge your bimanual robotic manipulation algorithms.  
See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

### Data Collection

We provide pre-collected trajectories within [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset). However, to leverage the configurability, we recommend collecting your data using the provided scripts:

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Task Configuration

Detailed configurations available in [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines

Several policy baselines are supported: [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html), [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html)

Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

Coming Soon.

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

**Benchmarking Generalizable Bimanual Manipulation:**
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

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos

For questions and suggestions, contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This repository is released under the [MIT license](./LICENSE).