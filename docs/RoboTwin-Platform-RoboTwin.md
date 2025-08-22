# RoboTwin: The Ultimate Bimanual Robotic Manipulation Platform

**Develop and benchmark cutting-edge bimanual robotic manipulation algorithms with RoboTwin, a comprehensive platform for research and development. Access the original repository at [https://github.com/RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin).**

## Key Features

*   **Scalable Data Generation:** Generate massive datasets with strong domain randomization for robust training.
*   **Comprehensive Benchmark:** Evaluate your algorithms against a diverse set of tasks and metrics.
*   **Dual-Arm Collaboration Challenge:** Participate in the CVPR 2025 MEIS Workshop challenge and push the boundaries of bimanual manipulation.
*   **Generative Digital Twins:** Leverage digital twins for realistic simulation and efficient training.
*   **Pre-collected Datasets:** Access over 100,000 pre-collected trajectories via Hugging Face.
*   **Diverse Policy Baselines:** Experiment with various policy baselines, including DP, ACT, DP3, RDT, PI0, OpenVLA-oft, TinyVLA, DexVLA and LLaVA-VLA.

## What's New in RoboTwin 2.0

RoboTwin 2.0 introduces significant advancements in bimanual robotic manipulation research:

*   **Leaderboard:** Explore performance metrics and compare algorithms on our updated [leaderboard](https://robotwin-platform.github.io/leaderboard).
*   **Enhanced Control Modes:** Utilize the new endpose control mode for increased precision.
*   **Fixed Bugs:** Addresses and resolves known issues for improved performance.
*   **Updated Technical Report:** Review the latest technical details of the RoboTwin Dual-Arm Collaboration Challenge [[arXiv](https://arxiv.org/abs/2506.23351)].
*   **Improved Domain Randomization:** Experience enhanced realism and robustness through domain randomization.

## Versions and Branches

*   **RoboTwin 2.0:** (Latest) - [main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main)
*   **RoboTwin 1.0:** - [1.0 Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0)
*   **RoboTwin 1.0 Code Generation:** - [1.0 Version GPT](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt)
*   **Early Version:** - [Early Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version)
*   **Challenge-Cup-2025** - [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025)
*   **CVPR-Challenge-2025-Round1** - [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1)
*   **CVPR-Challenge-2025-Round2** - [CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2)

## Installation

Install RoboTwin 2.0 by following the comprehensive instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes around 20 minutes.

## Tasks

Explore the diverse set of bimanual manipulation tasks offered by RoboTwin.  Refer to the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for comprehensive details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

### Document

For detailed information on how to use RoboTwin, please consult the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

Data collection is a crucial part of RoboTwin.  You can find pre-collected data via the [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset), but we encourage data generation.

<img src="./assets/files/domain_randomization.png" alt="Domain Randomization" style="display: block; margin: auto; width: 100%;">

#### Example: Running a Task

To run a task and collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Configuration

Customize the RoboTwin environment to your specifications using the detailed configuration instructions available in the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html).

## Policy Baselines

RoboTwin supports the following policy baselines for easy experimentation and evaluation:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   OpenVLA-oft
*   TinyVLA
*   DexVLA (Contributed by Media Group)
*   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))

Deploy your own policies using the [deploy guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## Experiment and Leaderboard

RoboTwin is ideal for exploring various research topics such as:

*   Single-task fine-tuning
*   Visual robustness
*   Language diversity robustness
*   Multi-task capability
*   Cross-embodiment performance

View the full leaderboard and settings: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Pre-collected Large-scale Dataset

Access and utilize the pre-collected datasets on Hugging Face: [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

## Citations

If you use RoboTwin in your research, please cite the following papers:

**RoboTwin 2.0**
```bibtex
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin**
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

**Benchmarking Generalizable Bimanual Manipulation**
```bibtex
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (Early Version)**
```bibtex
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

## Acknowledgements

**Software Support:** D-Robotics, **Hardware Support:** AgileX Robotics, **AIGC Support:** Deemos.

## Contact

For questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT license. See the [LICENSE](./LICENSE) file for details.