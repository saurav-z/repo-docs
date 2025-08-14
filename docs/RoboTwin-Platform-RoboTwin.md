# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

**Explore the cutting edge of dual-arm robotic manipulation with RoboTwin, a comprehensive platform for training, evaluating, and comparing robotic agents. [Go to the original repository](https://github.com/RoboTwin-Platform/RoboTwin)**

RoboTwin offers a powerful suite of tools and resources, including a data generator, benchmark tasks, and a vibrant community, to accelerate the development of advanced robotic manipulation skills.

## Key Features

*   **Comprehensive Benchmark:** Evaluate your bimanual robotic algorithms across a diverse set of challenging tasks.
*   **Scalable Data Generation:** Generate massive datasets with strong domain randomization to improve robustness.
*   **Multiple Versions and Branches:** Access different versions, including the latest RoboTwin 2.0, and specialized branches for challenges.
*   **Pre-collected Datasets:** Utilize over 100,000 pre-collected trajectories available on Hugging Face.
*   **Policy Baselines:** Experiment with and compare various policy baselines like DP, ACT, DP3, RDT, and PI0.
*   **Leaderboard:** Track your progress and compare your results against the community on the RoboTwin leaderboard.
*   **Community Support:** Engage with a growing community of researchers and developers.

## RoboTwin Versions

*   **RoboTwin 2.0 (Latest):** Features a scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
    *   Released: 2025/06/21
    *   Includes improvements and bug fixes.
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** Technical Report at CVPR 2025. [PDF](https://arxiv.org/pdf/2506.23351)
*   **RoboTwin 1.0:** A foundational benchmark for dual-arm robots with generative digital twins, *accepted to CVPR 2025 (Highlight)*. [PDF](https://arxiv.org/pdf/2504.13059)
*   **Early Version:** The initial release of RoboTwin, *received the Best Paper Award at the ECCV Workshop 2024*. [PDF](https://arxiv.org/pdf/2409.02920)

## Installation

Follow the detailed installation instructions in the [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).

## Tasks Overview

Explore a wide range of bimanual manipulation tasks.

See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

For detailed instructions on using the platform, refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

We strongly recommend users collect their own data due to the high configurability and diversity of task and embodiment setups.

<img src="./assets/files/domain_randomization.png" alt="Domain Randomization" style="display: block; margin: auto; width: 100%;">

### Running Tasks and Data Collection

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### Task Configuration

Consult the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for configuration details.

## Policy Baselines

RoboTwin provides various policy baselines for your evaluation, including:

*   DP
*   ACT
*   DP3
*   RDT
*   PI0
*   TinyVLA, DexVLA
*   LLaVA-VLA

Deploy Your Policy: [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

Experiment with different tasks, models, and settings.  Find the full leaderboard and settings at [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Citations

If you find RoboTwin helpful in your research, please cite the relevant papers:

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

Special thanks to:

*   **Software Support**: D-Robotics
*   **Hardware Support**: AgileX Robotics
*   **AIGC Support**: Deemos

Code Style: `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

## Contact

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is licensed under the MIT License.  See [LICENSE](./LICENSE) for details.