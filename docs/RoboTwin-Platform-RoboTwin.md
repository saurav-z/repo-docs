# RoboTwin: The Ultimate Benchmark for Bimanual Robotic Manipulation

**Unleash the potential of dual-arm robotics with RoboTwin, a cutting-edge platform for research and development.  Explore our [GitHub Repository](https://github.com/RoboTwin-Platform/RoboTwin) to get started!**

## Key Features

*   **Comprehensive Benchmark:** RoboTwin offers a diverse set of tasks, including manipulation, assembly, and object interaction, enabling thorough evaluation of robotic skills.
*   **Scalable Data Generation:**  Generate massive datasets with strong domain randomization to train robust and generalizable bimanual robotic systems.
*   **Robust Domain Randomization:**  Leverage advanced domain randomization techniques to enhance the robustness of your models against real-world variations.
*   **Multiple Policy Baselines:**  Includes support for various state-of-the-art policies like DP, ACT, DP3, RDT, PI0, TinyVLA, DexVLA, and LLaVA-VLA, offering a starting point for your research.
*   **Leaderboard & Evaluation:**  Track your progress and compare your models on our leaderboard ([https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard)).
*   **Active Community & Support:** Engage with a vibrant community through our documentation and updates to help you succeed.

## RoboTwin Versions & Releases

*   **RoboTwin 2.0 (Latest)**: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation ([Webpage](https://robotwin-platform.github.io/) | [Document](https://robotwin-platform.github.io/doc) | [Paper](https://arxiv.org/abs/2506.18088))
*   **RoboTwin Dual-Arm Collaboration Challenge @ CVPR'25 MEIS Workshop**: Technical Report ([PDF](https://arxiv.org/pdf/2506.23351) | [arXiv](https://arxiv.org/abs/2506.23351))
*   **RoboTwin 1.0**: Dual-Arm Robot Benchmark with Generative Digital Twins ([PDF](https://arxiv.org/pdf/2504.13059) | [arXiv](https://arxiv.org/abs/2504.13059))
*   **Early Version**:  Dual-Arm Robot Benchmark with Generative Digital Twins ([PDF](https://arxiv.org/pdf/2409.02920) | [arXiv](https://arxiv.org/abs/2409.02920))

## Updates

*   **2025/08/06**: RoboTwin 2.0 Leaderboard released ([leaderboard website](https://robotwin-platform.github.io/leaderboard)).
*   **2025/07/23**: RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
*   **2025/07/19**: Fixed DP3 evaluation code error.
*   **2025/07/09**: Updated endpose control mode ([RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)).
*   **2025/07/08**: Uploaded [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch.
*   **2025/07/02**: Fixed Piper Wrist Bug ([issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)).
*   **2025/07/01**: Released Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop.
*   **2025/06/21**: Released RoboTwin 2.0 ([Webpage](https://robotwin-platform.github.io/)).
*   **2025/04/11**: RoboTwin selected as <i>CVPR Highlight paper</i>!
*   **2025/02/27**: RoboTwin accepted to <i>CVPR 2025</i> !
*   **2024/09/30**: RoboTwin (Early Version) received <i>the Best Paper Award at the ECCV Workshop</i>!
*   **2024/09/20**: Officially released RoboTwin.

## Installation

See [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for detailed installation instructions. Installation takes approximately 20 minutes.

## Tasks

See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for detailed task information.

## Usage

Refer to [RoboTwin 2.0 Document (Usage)](https://robotwin-platform.github.io/doc/usage/index.html) for comprehensive usage details.

### Data Collection

We provide pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset). However, we recommend collecting your own data for maximum flexibility.

<img src="./assets/files/domain_randomization.png" alt="Domain Randomization Example" style="display: block; margin: auto; width: 100%;">

### 1. Task Running and Data Collection

Run the following command to collect data:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 2. Task Config

See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for details.

## Policy Baselines

### Policies Supported

*   DP ([DP Documentation](https://robotwin-platform.github.io/doc/usage/DP.html))
*   ACT ([ACT Documentation](https://robotwin-platform.github.io/doc/usage/ACT.html))
*   DP3 ([DP3 Documentation](https://robotwin-platform.github.io/doc/usage/DP3.html))
*   RDT ([RDT Documentation](https://robotwin-platform.github.io/doc/usage/RDT.html))
*   PI0 ([PI0 Documentation](https://robotwin-platform.github.io/doc/usage/Pi0.html))
*   TinyVLA ([TinyVLA Documentation](https://robotwin-platform.github.io/doc/usage/TinyVLA.html))
*   DexVLA ([DexVLA Documentation](https://robotwin-platform.github.io/doc/usage/DexVLA.html))
*   LLaVA-VLA ([LLaVA-VLA Documentation](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html))

### Deploy Your Policy

[Guide](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

⏰ TODO: G3Flow, HybridVLA, DexVLA, OpenVLA-OFT, SmolVLA, AVR, UniVLA

## Experiment & Leaderboard

We encourage exploration of the following topics with RoboTwin:

1.  Single-task fine-tuning
2.  Visual robustness
3.  Language diversity robustness (language condition)
4.  Multi-task capabilities
5.  Cross-embodiment performance

Find the full leaderboard and settings at: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

## Citations

Please cite the following if you use RoboTwin in your research:

**RoboTwin 2.0:**

```bibtex
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Liang, Qiwei and Li, Zixuan and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**RoboTwin:**

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

**Benchmarking Generalizable Bimanual Manipulation:**

```bibtex
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

**RoboTwin (Early Version):**

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

For questions or suggestions, contact [Tianxing Chen](https://tianxingchen.github.io).

## License

This project is released under the MIT license. See [LICENSE](./LICENSE) for details.