# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

RoboTwin is a cutting-edge platform designed to advance research in dual-arm robotic manipulation, offering a comprehensive benchmark with generative digital twins.  Explore the RoboTwin platform and contribute to the future of robotics!  Check out the original repo [here](https://github.com/RoboTwin-Platform/RoboTwin).

## Key Features

*   **Scalable Data Generation:** Generate diverse and realistic datasets for robust training.
*   **Strong Domain Randomization:**  Enhance generalization capabilities through robust domain randomization techniques.
*   **Comprehensive Benchmark:** Evaluate and compare algorithms across various bimanual manipulation tasks.
*   **Diverse Tasks:**  Supports a wide range of complex manipulation scenarios.
*   **Leaderboard:** Track and compare performance with state-of-the-art results on the RoboTwin leaderboard.
*   **Open Source:**  Freely accessible for research and development with an MIT license.

## Latest Updates

*   **RoboTwin 2.0:** The latest version features improvements and enhancements.
    *   **Webpage:** [https://robotwin-platform.github.io/](https://robotwin-platform.github.io/)
    *   **Document:** [https://robotwin-platform.github.io/doc/](https://robotwin-platform.github.io/doc/)
    *   **Paper:** [arXiv:2506.18088](https://arxiv.org/abs/2506.18088)
    *   **Leaderboard:** [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard)

*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR '25 MEIS Workshop:**
    *   **Technical Report:** [arXiv:2506.23351](https://arxiv.org/abs/2506.23351)

*   **RoboTwin 1.0 (CVPR 2025 Highlight):**
    *   **Paper:** [arXiv:2504.13059](https://arxiv.org/pdf/2504.13059)

*   **RoboTwin (Early Version, ECCV Workshop 2024 Best Paper Award):**
    *   **Paper:** [arXiv:2409.02920](https://arxiv.org/pdf/2409.02920)

## Overview

| Branch Name                                  | Link                                                                 |
| :------------------------------------------- | :------------------------------------------------------------------- |
| 2.0 Version Branch (latest)                  | [main](https://github.com/RoboTwin-Platform/RoboTwin/tree/main)         |
| 1.0 Version Branch                           | [1.0 Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0) |
| 1.0 Version Code Generation Branch           | [1.0 Version GPT](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt) |
| Early Version Branch                         | [Early Version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version) |
| 第十九届“挑战杯”人工智能专项赛分支                  | [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) |
| CVPR 2025 Challenge Round 1 Branch           | [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1) |
| CVPR 2025 Challenge Round 2 Branch           | [CVPR-Challenge-2025-Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2) |

## Installation

Follow the detailed instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html). Installation typically takes around 20 minutes.

## Tasks

Explore the wide range of bimanual manipulation tasks available in RoboTwin.  Refer to the [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for a comprehensive overview.

![RoboTwin Tasks](https://github.com/RoboTwin-Platform/RoboTwin/raw/main/assets/files/50_tasks.gif)

## Usage

Detailed usage instructions, including data collection and task configuration, are available in the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

We provide pre-collected trajectories for your convenience, but users are encouraged to collect their own data for maximum customization and task diversity.

*   **RoboTwin Dataset:** [https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset)

![Domain Randomization Example](https://github.com/RoboTwin-Platform/RoboTwin/raw/main/assets/files/domain_randomization.png)

### Task Running & Data Collection

Run the following command to collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Task Configuration

Consult the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for detailed task configuration options.

## Policy Baselines

RoboTwin supports a variety of policy baselines to facilitate your research:

*   [DP](https://robotwin-platform.github.io/doc/usage/DP.html)
*   [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
*   [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
*   [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
*   [PI0](https://robotwin-platform.github.io/doc/usage/Pi0.html)
*   [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
*   [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html) (Contributed by Media Group)
*   [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html) (Contributed by IRPN Lab, HKUST(GZ))

Deploy your own policies using the [deploy guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

## Experiment & Leaderboard

RoboTwin encourages experimentation in areas such as:

*   Single-task fine-tuning
*   Visual robustness
*   Language diversity robustness
*   Multi-task capabilities
*   Cross-embodiment performance

Find the latest leaderboard results and settings here: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard)

## Citations

If you utilize RoboTwin in your research, please cite the following papers:

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

## Acknowledgements

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

Code Style: `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

## Contact

For questions or suggestions, please reach out to [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.