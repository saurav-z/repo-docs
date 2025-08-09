# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

**Develop robust and generalizable bimanual robotic manipulation skills with the RoboTwin platform!**  Explore the latest advancements in robotic manipulation with RoboTwin, a comprehensive benchmark featuring cutting-edge data generation, strong domain randomization, and a competitive leaderboard. ([Original Repository](https://github.com/RoboTwin-Platform/RoboTwin))

## Key Features

*   **Comprehensive Benchmark:** Evaluate and compare your bimanual robotic manipulation algorithms across a diverse set of tasks.
*   **Scalable Data Generation:**  Generate realistic and varied training data with ease, enabling robust model training.
*   **Strong Domain Randomization:**  Enhance the generalizability of your models by leveraging powerful domain randomization techniques.
*   **Leaderboard & Community:**  Compete with others and track your progress on our public leaderboard. Connect with a vibrant community of researchers and practitioners.
*   **Multiple Versions & Challenges:** Access different versions of RoboTwin and participate in challenges, including the ongoing RoboTwin Dual-Arm Collaboration Challenge at CVPR'25 MEIS Workshop.

## Latest Updates

*   **RoboTwin 2.0 Released!**  Explore the latest version with enhanced features and improved performance.
    *   **Leaderboard:** Check out the [leaderboard website](https://robotwin-platform.github.io/leaderboard)
    *   **Recent Achievements:**  Outstanding Poster at ChinaSI 2025 (Ranking 1st),  Fix DP3 evaluation code error, update endpose control mode, and more!
    *   **Challenge Report:** Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)]

## Available Versions

*   **RoboTwin 2.0 (Latest):**  [Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088) | [arXiv](https://arxiv.org/abs/2506.18088) | [Leaderboard](https://robotwin-platform.github.io/leaderboard)
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop:** Official Technical Report: [PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351)
*   **RoboTwin 1.0:**  Accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>: [PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059)
*   **Early Version:**  Accepted to <i style="color: red; display: inline;"><b>ECCV Workshop 2024 (Best Paper Award)</b></i>: [PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920)

## Installation

Follow the detailed installation instructions in the [RoboTwin 2.0 Document](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).  Installation typically takes around 20 minutes.

## Tasks Overview

Explore the diverse range of manipulation tasks available in RoboTwin.  See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

## Usage

Detailed usage instructions are available in the [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html).

### Data Collection

Easily collect data for your experiments, including pre-collected datasets:  [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

<img src="./assets/files/domain_randomization.png" alt="Domain Randomization Example" style="display: block; margin: auto; width: 100%;">

### Running Tasks

1.  **Run Tasks & Data Collection:**

    ```bash
    bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
    # Example: bash collect_data.sh beat_block_hammer demo_randomized 0
    ```

2.  **Task Configuration:**

    Refer to the [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for customization options.

## Policy Baselines

*   **Supported Policies:**
    *   DP, ACT, DP3, RDT, PI0
    *   TinyVLA, DexVLA (Contributed by Media Group)
    *   LLaVA-VLA (Contributed by IRPN Lab, HKUST(GZ))
    *   And more to come! (G3Flow, HybridVLA, DexVLA, OpenVLA-OFT, SmolVLA, AVR, UniVLA - TODO)
*   **Deploy Your Policy:**  [guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

## Experiment & Leaderboard

RoboTwin is an excellent platform for exploring:

1.  Single-task fine-tuning capability
2.  Visual robustness
3.  Language diversity robustness (language condition)
4.  Multi-task capabilities
5.  Cross-embodiment performance

View the full leaderboard and settings at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Citations

Please cite the following papers if you use RoboTwin:

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

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos

Code Style: `find . -name "*.py" -exec sh -c 'echo "Processing: {}"; yapf -i --style='"'"'{based_on_style: pep8, column_limit: 120}'"'"' {}' \;`

For any questions or suggestions, please contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for details.