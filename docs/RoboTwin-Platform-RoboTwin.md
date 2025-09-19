# RoboTwin: Your Gateway to Advanced Bimanual Robotic Manipulation (v2.0!)

RoboTwin provides a comprehensive platform for research and development in bimanual robotic manipulation, featuring a scalable data generator, robust domain randomization, and a range of challenging tasks. Explore the RoboTwin platform on the original repository [here](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features:

*   **Scalable Data Generation:** Quickly generate diverse datasets for training and evaluating bimanual robotic manipulation models.
*   **Strong Domain Randomization:** Enhance the robustness of your models with realistic and varied simulated environments.
*   **Challenging Benchmark Tasks:** Test and compare your algorithms on a suite of complex bimanual manipulation tasks.
*   **Open-Source Baselines:** Leverage existing implementations of [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html) [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html), [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) and more to get started.
*   **Leaderboard:** Compare your results with state-of-the-art methods on our leaderboard: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## What's New in RoboTwin 2.0:

*   **[RoboTwin 2.0 Webpage](https://robotwin-platform.github.io/)**
*   **[RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc)**
*   **[RoboTwin 2.0 Paper](https://arxiv.org/abs/2506.18088)**

## Versions & Resources:

*   **RoboTwin 2.0 (Latest):** [Main Branch](https://github.com/RoboTwin-Platform/RoboTwin/tree/main)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR '25 MEIS Workshop:** [Technical Report](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0:** [1.0 Version Branch](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0)
*   **Early Version:** [Early Version Branch](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version)

## 🚀 Getting Started:

*   **Installation:** Follow the detailed installation instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation takes approximately 20 minutes.
*   **Tasks:** Explore the available tasks in the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html).
*   **Usage:** Refer to the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for detailed guidance on data collection, task configuration, and more.

## Data Collection and Task Running:

Collect data and begin your experiments with the following command:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

## Configuration:

*   See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for details.

## Policy Baselines:

*   **Supported Policies:**  [DP](https://robotwin-platform.github.io/doc/usage/DP.html), [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html), [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html), [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html), [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html), [OpenVLA-oft](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html), [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html), [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html), [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html).
*   **Deploy Your Policy:** [Guidance](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Pre-collected Dataset:

*   Access the pre-collected RoboTwin 2.0 dataset on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations:

Please cite the following publications if you use RoboTwin:

```
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

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

```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

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

Contact [Tianxing Chen](https://tianxingchen.github.io) with any questions or suggestions.

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.